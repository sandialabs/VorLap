#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#define QBLADE_EXPORT extern "C" __declspec(dllexport)
#else
#define QBLADE_EXPORT extern "C"
#ifndef __cdecl
#define __cdecl
#endif
#include <dlfcn.h>
#endif

#include <filesystem>

namespace {

constexpr int kMessageBufferSize = 1024;

std::mutex g_mutex;
PyObject* g_runtime = nullptr;
PyObject* g_update_callable = nullptr;
PyObject* g_message_callable = nullptr;
std::string g_message = "VorLap bridge has not been initialized.";
std::string g_bridge_log_path;
npy_intp g_swap_size = 0;
bool g_python_ready = false;
unsigned long long g_update_counter = 0;

void append_bridge_log_line(const std::string& message) {
    if (g_bridge_log_path.empty()) {
        return;
    }
    std::ofstream log(g_bridge_log_path, std::ios::app);
    if (!log) {
        return;
    }
    log << message << '\n';
}

void init_bridge_log_path(const char* param_file) {
    namespace fs = std::filesystem;
    try {
        fs::path base_dir;
        if (param_file && std::strlen(param_file) > 0) {
            fs::path param_path(param_file);
            base_dir = param_path.has_parent_path() ? param_path.parent_path() : fs::current_path();
        } else {
            base_dir = fs::current_path();
        }
        fs::create_directories(base_dir);
        g_bridge_log_path = (base_dir / "vorlap_qblade_bridge_cpp.log").string();
    } catch (...) {
        g_bridge_log_path.clear();
    }
}

void set_message(const std::string& message) {
    g_message = message;
    if (g_message.size() >= static_cast<size_t>(kMessageBufferSize)) {
        g_message.resize(kMessageBufferSize - 1);
    }
    append_bridge_log_line("MESSAGE: " + g_message);
}

std::string fetch_python_error() {
    PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    std::string message = "unknown Python error";
    if (ptype) {
        PyObject* type_name_obj = PyObject_GetAttrString(ptype, "__name__");
        if (type_name_obj) {
            const char* type_name = PyUnicode_AsUTF8(type_name_obj);
            if (type_name) {
                message = std::string(type_name) + ": ";
            }
            Py_DECREF(type_name_obj);
        }
    }
    if (pvalue) {
        PyObject* value_str = PyObject_Str(pvalue);
        if (value_str) {
            const char* value_cstr = PyUnicode_AsUTF8(value_str);
            if (value_cstr) {
                message += value_cstr;
            }
            Py_DECREF(value_str);
        }
    }

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    return message;
}

bool ensure_numpy_ready() {
    import_array1(false);
    return true;
}

bool ensure_python_library_global() {
#if defined(_WIN32) || defined(_WIN64)
    return true;
#else
    Dl_info info;
    std::memset(&info, 0, sizeof(info));
    if (dladdr(reinterpret_cast<void*>(Py_Initialize), &info) == 0 || !info.dli_fname) {
        set_message("VorLap bridge could not locate the loaded libpython for RTLD_GLOBAL promotion.");
        return false;
    }

    void* handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char* err = dlerror();
        set_message(
            std::string("VorLap bridge failed to promote libpython to RTLD_GLOBAL: ")
            + (err ? err : "unknown dlopen error")
        );
        return false;
    }
    return true;
#endif
}

bool ensure_python_initialized() {
    if (g_python_ready) {
        return true;
    }
    if (!ensure_python_library_global()) {
        return false;
    }
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    if (!Py_IsInitialized()) {
        set_message("VorLap bridge failed to initialize the embedded Python interpreter.");
        return false;
    }
    return true;
}

void clear_runtime_locked() {
    Py_XDECREF(g_update_callable);
    Py_XDECREF(g_message_callable);
    Py_XDECREF(g_runtime);
    g_update_callable = nullptr;
    g_message_callable = nullptr;
    g_runtime = nullptr;
    g_swap_size = 0;
    g_update_counter = 0;
}

bool store_runtime_metadata_locked() {
    PyObject* size_obj = PyObject_GetAttrString(g_runtime, "swap_size");
    if (!size_obj) {
        set_message("VorLap runtime object does not expose swap_size.");
        return false;
    }
    long long swap_size = PyLong_AsLongLong(size_obj);
    Py_DECREF(size_obj);
    if (swap_size <= 0 || PyErr_Occurred()) {
        set_message("VorLap runtime returned an invalid swap_size.");
        return false;
    }
    g_swap_size = static_cast<npy_intp>(swap_size);

    g_update_callable = PyObject_GetAttrString(g_runtime, "update");
    if (!g_update_callable) {
        set_message("VorLap runtime object does not expose update().");
        return false;
    }
    if (!PyCallable_Check(g_update_callable)) {
        set_message("VorLap runtime update attribute is not callable.");
        return false;
    }

    g_message_callable = PyObject_GetAttrString(g_runtime, "update_message");
    if (!g_message_callable) {
        PyErr_Clear();
    } else if (!PyCallable_Check(g_message_callable)) {
        Py_DECREF(g_message_callable);
        g_message_callable = nullptr;
    }
    return true;
}

bool initialize_runtime_locked(const char* param_file) {
    if (!ensure_python_initialized()) {
        return false;
    }

    PyObject* module = PyImport_ImportModule("vorlap.qblade_runtime");
    if (!module) {
        set_message("Could not import vorlap.qblade_runtime: " + fetch_python_error());
        return false;
    }

    PyObject* factory = PyObject_GetAttrString(module, "create_runtime");
    if (!factory || !PyCallable_Check(factory)) {
        Py_XDECREF(factory);
        Py_DECREF(module);
        set_message("vorlap.qblade_runtime.create_runtime is missing or not callable.");
        return false;
    }

    PyObject* param_arg = PyUnicode_FromString(param_file ? param_file : "");
    if (!param_arg) {
        Py_DECREF(factory);
        Py_DECREF(module);
        set_message("Could not create Python string for the parameter file path.");
        return false;
    }

    PyObject* runtime = PyObject_CallFunctionObjArgs(factory, param_arg, nullptr);
    Py_DECREF(param_arg);
    Py_DECREF(factory);
    Py_DECREF(module);
    if (!runtime) {
        set_message("Failed to create VorLap QBlade runtime: " + fetch_python_error());
        return false;
    }

    clear_runtime_locked();
    g_runtime = runtime;
    if (!store_runtime_metadata_locked()) {
        clear_runtime_locked();
        return false;
    }

    set_message("VorLap QBlade runtime initialized.");
    return true;
}

PyObject* make_swap_array_view(float* avr_swap) {
    if (g_swap_size <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "VorLap runtime swap size is not initialized.");
        return nullptr;
    }
    npy_intp dims[1] = {g_swap_size};
    PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, reinterpret_cast<void*>(avr_swap));
    if (!array) {
        return nullptr;
    }
    return array;
}

bool refresh_message_from_runtime_locked() {
    if (!g_message_callable) {
        return true;
    }
    PyObject* result = PyObject_CallNoArgs(g_message_callable);
    if (!result) {
        set_message("VorLap runtime update_message() failed: " + fetch_python_error());
        return false;
    }
    const char* message_text = PyUnicode_AsUTF8(result);
    if (!message_text) {
        Py_DECREF(result);
        set_message("VorLap runtime update_message() did not return a string.");
        return false;
    }
    set_message(message_text);
    Py_DECREF(result);
    return true;
}

void copy_message_to_buffer(char* out_buf) {
    if (!out_buf) {
        return;
    }
    std::memset(out_buf, 0, kMessageBufferSize);
    std::strncpy(out_buf, g_message.c_str(), kMessageBufferSize - 1);
}

}  // namespace

QBLADE_EXPORT void __cdecl update_init(const char* paramFile) {
    std::lock_guard<std::mutex> guard(g_mutex);
    init_bridge_log_path(paramFile);
    append_bridge_log_line(
        std::string("update_init entered; paramFile=") + (paramFile ? paramFile : "<null>")
    );
    if (!ensure_python_initialized()) {
        return;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    if (!g_python_ready) {
        if (!ensure_numpy_ready()) {
            set_message("VorLap bridge failed to initialize the NumPy C API.");
            PyGILState_Release(gil);
            return;
        }
        g_python_ready = true;
    }
    if (!initialize_runtime_locked(paramFile)) {
        if (!PyErr_Occurred()) {
            PyErr_Clear();
        }
    }
    append_bridge_log_line("update_init exit; message=" + g_message);
    PyGILState_Release(gil);
}

QBLADE_EXPORT void __cdecl update(float* avrSwap) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_update_counter;
    const bool log_this_update = (g_update_counter <= 10) || (g_update_counter % 100 == 0);
    if (log_this_update) {
        append_bridge_log_line("update entered; count=" + std::to_string(g_update_counter));
    }
    if (!g_runtime || !g_update_callable) {
        set_message("VorLap runtime is not initialized. QBlade must call update_init() first.");
        return;
    }

    PyGILState_STATE gil = PyGILState_Ensure();

    PyObject* swap_view = make_swap_array_view(avrSwap);
    if (!swap_view) {
        set_message("VorLap bridge could not build a NumPy view over avrSwap: " + fetch_python_error());
        PyErr_Clear();
        PyGILState_Release(gil);
        return;
    }

    PyObject* result = PyObject_CallFunctionObjArgs(g_update_callable, swap_view, nullptr);
    Py_DECREF(swap_view);
    if (!result) {
        set_message("VorLap runtime update() failed: " + fetch_python_error());
        PyErr_Clear();
        PyGILState_Release(gil);
        return;
    }
    Py_DECREF(result);

    if (!refresh_message_from_runtime_locked()) {
        PyErr_Clear();
    }
    if (log_this_update) {
        append_bridge_log_line("update exit; count=" + std::to_string(g_update_counter) + "; message=" + g_message);
    }

    PyGILState_Release(gil);
}

QBLADE_EXPORT void __cdecl update_message(char* outBuf) {
    std::lock_guard<std::mutex> guard(g_mutex);
    copy_message_to_buffer(outBuf);
}

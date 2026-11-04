#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define QBLADE_EXPORT extern "C" __declspec(dllexport)
#else
#define QBLADE_EXPORT extern "C"
#ifndef __cdecl
#define __cdecl
#endif
#include <dlfcn.h>
#endif

namespace {

constexpr int kMessageBufferSize = 1024;

std::mutex g_mutex;
PyObject* g_runtime = nullptr;
PyObject* g_update_callable = nullptr;
PyObject* g_message_callable = nullptr;
PyObject* g_clear_callable = nullptr;
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

std::string parent_dir_from_path(const char* path) {
    if (!path || !*path) {
        return ".";
    }
    std::string full(path);
    const std::string::size_type pos = full.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return full.substr(0, 1);
    }
    return full.substr(0, pos);
}

std::string extract_json_string_field(const char* json_path, const std::string& key) {
    if (!json_path || !*json_path) {
        return "";
    }
    std::ifstream input(json_path);
    if (!input) {
        return "";
    }
    std::string text;
    std::string line;
    while (std::getline(input, line)) {
        text += line;
        text.push_back('\n');
    }

    const std::string needle = "\"" + key + "\"";
    const std::string::size_type key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return "";
    }
    const std::string::size_type colon_pos = text.find(':', key_pos + needle.size());
    if (colon_pos == std::string::npos) {
        return "";
    }
    const std::string::size_type quote_pos = text.find('"', colon_pos + 1);
    if (quote_pos == std::string::npos) {
        return "";
    }

    std::string value;
    bool escaped = false;
    for (std::string::size_type i = quote_pos + 1; i < text.size(); ++i) {
        const char ch = text[i];
        if (escaped) {
            switch (ch) {
                case '"':
                case '\\':
                case '/':
                    value.push_back(ch);
                    break;
                case 'n':
                    value.push_back('\n');
                    break;
                case 'r':
                    value.push_back('\r');
                    break;
                case 't':
                    value.push_back('\t');
                    break;
                default:
                    value.push_back(ch);
                    break;
            }
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return value;
        }
        value.push_back(ch);
    }
    return "";
}

void init_bridge_log_path(const char* param_file) {
    std::string base_dir;
    const char* env_log_dir = std::getenv("VORLAP_QBLADE_BRIDGE_LOG_DIR");
    if (env_log_dir && *env_log_dir) {
        base_dir = env_log_dir;
    } else {
        base_dir = extract_json_string_field(param_file, "source_parameter_dir");
    }
    if (base_dir.empty()) {
        base_dir = parent_dir_from_path(param_file);
    }
    if (base_dir.empty()) {
        g_bridge_log_path = "vorlap_qblade_bridge_cpp.log";
        return;
    }
    const char sep = (base_dir.back() == '/' || base_dir.back() == '\\') ? '\0' : '/';
    g_bridge_log_path = base_dir;
    if (sep != '\0') {
        g_bridge_log_path.push_back(sep);
    }
    g_bridge_log_path += "vorlap_qblade_bridge_cpp.log";
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

const char* kConfigurePythonPathSource = R"PY(
def _vorlap_bridge_configure_python_path(param_file):
    import json
    import os
    import sys

    report = {
        "sys_executable": getattr(sys, "executable", ""),
        "python_version": sys.version,
        "sys_prefix": getattr(sys, "prefix", ""),
    }
    try:
        param_path = os.path.abspath(param_file) if param_file else ""
        base_dir = os.path.dirname(param_path) if param_path else os.getcwd()
        with open(param_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        raw_entries = config.get("python_path_entries", config.get("python_package_path", []))
        if raw_entries is None:
            raw_entries = []
        if isinstance(raw_entries, (str, bytes, os.PathLike)):
            raw_entries = [raw_entries]

        source_parameter_dir = config.get("source_parameter_dir")
        source_base = None
        if source_parameter_dir not in (None, ""):
            source_text = os.fspath(source_parameter_dir)
            if os.path.isabs(source_text):
                source_base = os.path.normpath(source_text)
            else:
                source_base = os.path.normpath(os.path.join(base_dir, source_text))

        resolved_entries = []
        for raw_entry in raw_entries:
            if raw_entry in (None, ""):
                continue
            entry_text = os.fspath(raw_entry)
            candidates = []
            if os.path.isabs(entry_text):
                candidates.append(os.path.normpath(entry_text))
            else:
                candidates.append(os.path.normpath(os.path.join(base_dir, entry_text)))
                if source_base:
                    candidates.append(os.path.normpath(os.path.join(source_base, entry_text)))
            selected = next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
            resolved_entries.append(selected)

        added_entries = []
        for entry in reversed(resolved_entries):
            if entry and entry not in sys.path:
                sys.path.insert(0, entry)
                added_entries.insert(0, entry)

        report["python_path_entries"] = resolved_entries
        report["python_path_entries_added"] = added_entries
    except Exception as exc:
        report["config_error"] = f"{type(exc).__name__}: {exc}"

    report["sys_path"] = list(sys.path)
    return json.dumps(report, sort_keys=True)
)PY";

const char* kRuntimeDiagnosticsSource = R"PY(
def _vorlap_bridge_runtime_diagnostics(_unused):
    import importlib
    import json
    import sys

    report = {
        "sys_executable": getattr(sys, "executable", ""),
        "python_version": sys.version,
        "sys_prefix": getattr(sys, "prefix", ""),
        "sys_path": list(sys.path),
    }
    for module_name in ("numpy", "vorlap", "vorlap.qblade_runtime"):
        key = module_name.replace(".", "_")
        try:
            module = importlib.import_module(module_name)
            report[f"{key}_file"] = getattr(module, "__file__", "")
            version = getattr(module, "__version__", None)
            if version is not None:
                report[f"{key}_version"] = str(version)
        except Exception as exc:
            report[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    return json.dumps(report, sort_keys=True)
)PY";

std::string call_python_string_helper(
    const char* source,
    const char* function_name,
    const char* argument
) {
    PyObject* globals = PyDict_New();
    if (!globals) {
        return "could not allocate Python helper globals";
    }
    PyObject* builtins = PyEval_GetBuiltins();
    if (builtins) {
        PyDict_SetItemString(globals, "__builtins__", builtins);
    }

    PyObject* exec_result = PyRun_String(source, Py_file_input, globals, globals);
    if (!exec_result) {
        std::string error = fetch_python_error();
        Py_DECREF(globals);
        return "helper definition failed: " + error;
    }
    Py_DECREF(exec_result);

    PyObject* helper = PyDict_GetItemString(globals, function_name);
    if (!helper || !PyCallable_Check(helper)) {
        Py_DECREF(globals);
        return std::string("helper function missing: ") + function_name;
    }

    PyObject* arg = PyUnicode_FromString(argument ? argument : "");
    if (!arg) {
        std::string error = fetch_python_error();
        Py_DECREF(globals);
        return "helper argument creation failed: " + error;
    }

    PyObject* result = PyObject_CallFunctionObjArgs(helper, arg, nullptr);
    Py_DECREF(arg);
    if (!result) {
        std::string error = fetch_python_error();
        Py_DECREF(globals);
        return "helper call failed: " + error;
    }

    const char* text = PyUnicode_AsUTF8(result);
    std::string rendered = text ? text : "<non-UTF8 helper result>";
    Py_DECREF(result);
    Py_DECREF(globals);
    return rendered;
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
    std::vector<std::string> candidates;
    if (dladdr(reinterpret_cast<void*>(Py_Initialize), &info) != 0 && info.dli_fname) {
        candidates.emplace_back(info.dli_fname);
    }

    const std::string version =
        std::to_string(PY_MAJOR_VERSION) + "." + std::to_string(PY_MINOR_VERSION);
    candidates.emplace_back("libpython" + version + ".so.1.0");
    candidates.emplace_back("libpython" + version + ".so");

    std::string errors;
    for (const std::string& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        dlerror();
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            return true;
        }
        const char* err = dlerror();
        errors += candidate + ": " + (err ? err : "unknown dlopen error") + "; ";
    }

    set_message("VorLap bridge failed to promote libpython to RTLD_GLOBAL: " + errors);
    return false;
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
    Py_XDECREF(g_clear_callable);
    Py_XDECREF(g_runtime);
    g_update_callable = nullptr;
    g_message_callable = nullptr;
    g_clear_callable = nullptr;
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

    g_clear_callable = PyObject_GetAttrString(g_runtime, "clear_force_output");
    if (!g_clear_callable) {
        set_message("VorLap runtime object does not expose clear_force_output().");
        return false;
    }
    if (!PyCallable_Check(g_clear_callable)) {
        set_message("VorLap runtime clear_force_output attribute is not callable.");
        return false;
    }
    return true;
}

bool initialize_runtime_locked(const char* param_file) {
    if (!ensure_python_initialized()) {
        return false;
    }

    append_bridge_log_line(
        "python_env_before_import="
        + call_python_string_helper(kConfigurePythonPathSource, "_vorlap_bridge_configure_python_path", param_file)
    );

    PyObject* module = PyImport_ImportModule("vorlap.qblade_runtime");
    if (!module) {
        set_message("Could not import vorlap.qblade_runtime: " + fetch_python_error());
        return false;
    }
    append_bridge_log_line(
        "python_env_after_import="
        + call_python_string_helper(kRuntimeDiagnosticsSource, "_vorlap_bridge_runtime_diagnostics", "")
    );

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

bool clear_force_output_locked(PyObject* swap_view, const std::string& message) {
    if (!g_clear_callable) {
        set_message("VorLap runtime update() failed before clear_force_output() was available: " + message);
        return false;
    }

    PyObject* message_arg = PyUnicode_FromString(message.c_str());
    if (!message_arg) {
        set_message("VorLap bridge could not create clear-force message: " + fetch_python_error());
        return false;
    }

    PyObject* result = PyObject_CallFunctionObjArgs(g_clear_callable, swap_view, message_arg, nullptr);
    Py_DECREF(message_arg);
    if (!result) {
        set_message("VorLap runtime clear_force_output() failed after update error: " + fetch_python_error());
        return false;
    }
    Py_DECREF(result);

    return refresh_message_from_runtime_locked();
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
        if (PyErr_Occurred()) {
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
    if (!avrSwap) {
        set_message("VorLap bridge received a null avrSwap pointer; update skipped.");
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
    if (!result) {
        const std::string update_error = fetch_python_error();
        PyErr_Clear();
        clear_force_output_locked(
            swap_view,
            "VorLap runtime update() failed: " + update_error + "; force output cleared."
        );
        Py_DECREF(swap_view);
        PyGILState_Release(gil);
        return;
    }
    Py_DECREF(swap_view);
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

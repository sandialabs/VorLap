cmake_minimum_required(VERSION 3.18)

project(vorlap_qblade_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Development NumPy)

add_library(vorlap_qblade_bridge SHARED src/vorlap_qblade_bridge.cpp)

target_compile_definitions(vorlap_qblade_bridge PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_include_directories(
    vorlap_qblade_bridge
    PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(vorlap_qblade_bridge PRIVATE Python3::Python)

set_target_properties(
    vorlap_qblade_bridge
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "libvorlap_qblade_bridge"
        POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
    target_compile_options(vorlap_qblade_bridge PRIVATE /utf-8)
endif()


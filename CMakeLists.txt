cmake_minimum_required(VERSION 3.0)
project(
    "C++ ML"
    VERSION 0.1.0
    DESCRIPTION "Machine learning models implemented in c++"
    LANGUAGES CXX
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(SRC
    src/hmm.cpp
)
add_library(cpp_ml STATIC ${SRC})
target_include_directories(cpp_ml
    PUBLIC include
)
target_link_libraries(cpp_ml
    PUBLIC Eigen3::Eigen
)

add_subdirectory(examples)

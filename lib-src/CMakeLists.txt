cmake_minimum_required(VERSION 3.27)
project(lib_src)
set(CMAKE_CXX_STANDARD 17)


# TODO WIP: For now we just create an executable, but we should create a library
add_executable(lib_src
        main.cpp
        Models/Linear.cpp
        Models/Linear.hpp
        Models/MLP.cpp
        Models/MLP.hpp
)

target_link_libraries(lib_src)

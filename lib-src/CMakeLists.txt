cmake_minimum_required(VERSION 3.27)
project(lib_src)

set(CMAKE_CXX_STANDARD 17)

add_library(lib_src STATIC library.cpp)

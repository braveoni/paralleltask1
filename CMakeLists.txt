cmake_minimum_required(VERSION 3.21)
project(paralleltask1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")

add_executable(paralleltask1 main.cpp)

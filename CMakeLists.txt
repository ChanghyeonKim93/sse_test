cmake_minimum_required(VERSION 2.8.3)
project(sse_test)
find_package(cmake_modules)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

set(SRC ${SRC} main.cpp)

include_directories( ${includeDirs} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O2 -m64 -Ofast -flto -msse3 -march=native  -mfpmath=sse")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -m64 -Ofast -flto -msse3 -march=native -mfpmath=sse")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -flto -pg -march=native -mfpmath=sse")
set(DCMAKE_build_type debug)


add_executable (sse_test ${SRC} )

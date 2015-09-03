cmake_minimum_required(VERSION 2.8)
project(pom)

set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS} -std=c++14 -O3 -march=native")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

set(SOURCE_FILES
  main.cpp
  pom.h)

add_executable(pom main.cpp)
target_link_libraries(pom ${OpenCV_LIBS} armadillo)


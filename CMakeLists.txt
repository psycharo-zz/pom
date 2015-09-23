cmake_minimum_required(VERSION 2.8)
project(pom)

set(CMAKE_CXX_FLAGS "-pg ${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(SOURCE_FILES
  main.cpp
  pom.h)

add_executable(pom main.cpp)
target_link_libraries(pom ${OpenCV_LIBS} armadillo)


cmake_minimum_required(VERSION 3.10)
project(corridor_gen)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common octree filters)

add_library(${PROJECT_NAME} src/corridor_gen.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
)
cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)


project(run)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(run detect2.cpp)

target_link_libraries(run ${OpenCV_LIBS})

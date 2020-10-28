cmake_minimum_required(VERSION 3.17)
project(torch_test)

set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(torch_test main.cpp)
target_link_libraries(torch_test "${TORCH_LIBRARIES}")
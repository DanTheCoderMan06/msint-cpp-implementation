cmake_minimum_required(VERSION 3.25)
project(Rooster LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(Eigen)

set(EIGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

add_executable(rooster main.cpp NeuralNetwork.cpp utils.cpp)
target_link_libraries(rooster PRIVATE Eigen3::Eigen)
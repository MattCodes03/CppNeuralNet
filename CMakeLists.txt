cmake_minimum_required (VERSION 3.12)

project("NueralNetwork")


file(GLOB_RECURSE SOURCES
    src/*.h
    src/*.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/src)


add_executable(NueralNetwork ${SOURCES})


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NueralNetwork PROPERTY CXX_STANDARD 20)
endif()

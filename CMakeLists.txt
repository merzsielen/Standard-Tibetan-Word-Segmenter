cmake_minimum_required (VERSION 3.22)

project (lhtnnt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BASE_SRCS
    "src/neural/neuralnet.cpp"
    "src/neural/neuralnet.h"
    "src/utility/filehandling.h"
    "src/utility/scrubber.cpp"
    "src/utility/scrubber.h"
    "src/main.cpp"
)

add_executable (lhtnnt ${BASE_SRCS})

add_custom_target(copy_datasets
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/copy-datasets.cmake
)

add_dependencies(lhtnnt copy_datasets)

add_subdirectory(libs/armadillo-12.6.2)

target_link_libraries(lhtnnt armadillo)
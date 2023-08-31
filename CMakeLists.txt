cmake_minimum_required (VERSION 3.22)

project (lhtnnws LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BASE_SRCS
    "src/neural/datahandler.cpp"
    "src/neural/datahandler.h"
    "src/neural/neuralnet.cpp"
    "src/neural/neuralnet.h"
    "src/utility/filehandling.cpp"
    "src/utility/filehandling.h"
    "src/utility/scrubber.cpp"
    "src/utility/scrubber.h"
    "src/main.cpp"
)

add_executable (lhtnnws ${BASE_SRCS})

add_custom_target(copy_datasets
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/copy-datasets.cmake
)

add_dependencies(lhtnnws copy_datasets)

add_subdirectory(libs/armadillo-12.6.2)
add_subdirectory(libs/csv-parser)

target_link_libraries(lhtnnws armadillo csv)
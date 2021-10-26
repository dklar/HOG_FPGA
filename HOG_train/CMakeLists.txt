cmake_minimum_required(VERSION 3.0.0)
project(HOG_train VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED PATHS "/usr/local" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(HOG_train main.cpp hog.hpp)
target_link_libraries( HOG_train ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

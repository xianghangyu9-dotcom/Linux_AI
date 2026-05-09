cmake_minimum_required(VERSION 3.10)

project(app)

#C++11标准
set(CMAKE_CXX_STANDARD 11)

#找包
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
#输出库信息确保正常
message(STATUS "OpenCV status:")
message(STATUS "    version:${OpenCV_VERSION}")
message(STATUS "    libraries:${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(LIB_ARCH "aarch64")
set(RGA_PATH "rga")
set(RGA_AARCH "lib")
set(3RDPARTY_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RKNN_API_PATH       ${3RDPARTY_PATH}/librknn_api)
set(RKNN_INCLUDE_DIRS   ${RKNN_API_PATH}/include)
set(RKNN_LIBS           ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
set(RGA_DIRS            ${3RDPARTY_PATH}/${RGA_PATH}/include)
set(RGA_LIBS            ${3RDPARTY_PATH}/${RGA_PATH}/${RGA_AARCH}/librga.so)

#包含头文件INCLUDE_DIRS
include_directories(${OpenCV_INCLUDE_DIRS} ${RKNN_INCLUDE_DIRS} ${RGA_DIRS} ${LIBV4L2_INCLUDE_DIRS})

add_executable(app videodeal.cpp thread_pool.cpp yolov5s.cpp process.cpp)

#链接库LIBS
target_link_libraries(app ${OpenCV_LIBS} Threads::Threads ${RKNN_LIBS} ${RGA_LIBS} ${LIBV4L2_LIBRARIES})
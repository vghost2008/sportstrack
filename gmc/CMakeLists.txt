cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gmc)


find_package(OpenCV REQUIRED)

# Define our library target
add_executable(gmc main.cpp)
# Enable C++14
target_compile_features(gmc PRIVATE cxx_std_14)
# Link against LibTorch
# Link against OpenCV
target_link_libraries(gmc opencv_core opencv_imgproc opencv_calib3d opencv_imgproc opencv_core opencv_ml opencv_features2d opencv_objdetect opencv_flann opencv_video opencv_highgui opencv_videostab pthread stdc++fs)

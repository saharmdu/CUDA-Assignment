CUDA Image Processing Assignment:
This repository contains CUDA-based image processing executables, demonstrating various operations on images using CUDA and OpenCV. The project is structured to apply different image processing techniques, including grayscale conversion, combining images, and processing with multiple streams for efficiency.
The output of graysacale is included in data folder.

Prerequisites
To build and run these examples, you will need:

CUDA Toolkit (compatible with your GPU)
OpenCV (tested with version 3.x or 4.x)
CMake (version 2.8 or higher)
A compatible C++ compiler
Ensure that CUDA and OpenCV are correctly installed on your system and that the environment variables are properly set up for CUDA and OpenCV libraries.

Building the Project
Clone the repository to your local machine.
Navigate to the project directory.
Create a build directory and navigate into it:
bash
Copy code
mkdir build && cd build
Run CMake to configure the project:
Copy code
cmake ..
Build the project:
go
Copy code
make
This will compile the executables listed in the CMakeLists.txt file.

Executables
The project compiles the following CUDA executables:

struct_test: Demonstrates using structures and pointers in CUDA.
struct_array: Shows handling arrays of structures in CUDA.
combine_gray: Combines multiple images into a single grayscale image.
combine_rgb: Combines multiple images into a single RGB image.
gray: Converts RGB images to grayscale.
multi_stream_gray: Utilizes CUDA streams for efficient grayscale conversion of multiple images concurrently.

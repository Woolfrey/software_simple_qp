cmake_minimum_required(VERSION 3.8)					                 # Minimum cmake requirement

project(SimpleQPSolver VERSION 1.0.0 LANGUAGES CXX)                                      # Can use ${PROJECT_NAME} as a reference

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

# Default build is Release. To build in other modes use:
#	- Debug			(Information, not optimised)
#	- Release		(No information, optimised)
#	- RelWithDebInfo	(Information & optimised)
#	- MinSizeRel		(Same as Release, but optimised for size)
#
# Then run `make` as usual.

find_package(Eigen3 REQUIRED)                                                            # Eigen libraries must be installed

include_directories(include)                                                             # Location of header files

add_executable(test src/test.cpp)                                                        # Declare location of source files for executable

target_link_libraries(test Eigen3::Eigen)                                                # Link Eigen libraries to the executable

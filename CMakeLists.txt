cmake_minimum_required(VERSION 3.0.2)
project(gazebo_px4_simulator)

## Compile as C++11 (or newer if needed)
add_compile_options(-std=c++11)

## Find required packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  gazebo_ros
  geometry_msgs
  message_generation
)

find_package(gazebo REQUIRED)
find_package(Protobuf REQUIRED)

# Add message files
add_message_files(
  FILES
  OdorReading.msg
)

# Python setup must come before generate_messages
catkin_python_setup()

# Generate messages must come after catkin_python_setup but before catkin_package
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Declare catkin package after generate_messages
catkin_package(
  CATKIN_DEPENDS rospy roscpp geometry_msgs message_runtime
)

## Set include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Link directories
link_directories(${GAZEBO_LIBRARY_DIRS})


## Install headers and proto files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


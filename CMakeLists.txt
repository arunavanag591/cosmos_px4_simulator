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
)

find_package(gazebo REQUIRED)
find_package(Protobuf REQUIRED)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS rospy roscpp geometry_msgs
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

## Define the path to the .proto files
set(PROTO_FILES
  proto/Wind.proto
  proto/vector3d.proto
)

## Generate Protobuf files
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Add -fPIC to the Protobuf-generated library
add_library(physics_msgs SHARED ${PROTO_SRCS} ${PROTO_HDRS})
set_target_properties(physics_msgs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(physics_msgs ${PROTOBUF_LIBRARIES})

## Build the WindPublisherPlugin
add_library(WindPublisherPlugin SHARED src/WindPublisherPlugin.cpp)
target_link_libraries(WindPublisherPlugin
  physics_msgs
  ${GAZEBO_LIBRARIES}
  ${catkin_LIBRARIES}
)

## Build the WindSubscriberTest executable
add_executable(wind_subscriber_test src/WindSubscriberTest.cpp)
target_link_libraries(wind_subscriber_test
  physics_msgs
  ${GAZEBO_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${catkin_LIBRARIES}
)

## Install the plugin
install(TARGETS WindPublisherPlugin physics_msgs
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## Install the subscriber executable
install(TARGETS wind_subscriber_test
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install headers and proto files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES
  proto/Wind.proto
  proto/vector3d.proto
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/proto
)

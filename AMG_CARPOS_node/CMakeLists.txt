cmake_minimum_required(VERSION 3.0.2)
project(CARPOS_amg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  amg_message.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


install(PROGRAMS
  scripts/carpos_amg_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


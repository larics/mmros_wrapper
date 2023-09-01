cmake_minimum_required(VERSION 3.0.2)
project(mmros_wrapper)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  mmros_utils
)

catkin_package(
  CATKIN_DEPENDS rospy roscpp std_msgs sensor_msgs cv_bridge mmros_utils
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/mmros_wrapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

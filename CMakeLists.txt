cmake_minimum_required(VERSION 3.0.2)
project(mmros_wrapper)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy roscpp std_msgs sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/mmros_wrapper.py
  scripts/mmros_wrapper_ckpt.py
  scripts/shm_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

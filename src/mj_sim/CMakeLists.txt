cmake_minimum_required(VERSION 3.8)
project(mj_sim)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0072 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# custom message
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LowState.msg"
  "msg/UpCmd.msg"
  DEPENDENCIES std_msgs
)

include_directories(include)

add_library(xbox_controller SHARED src/xbox_controller.cpp)
add_library(ik_solver SHARED src/ik_solver.cpp)
add_library(gait src/gait.cpp)
add_executable(robot_ctrl src/robot_ctrl.cpp)

rosidl_get_typesupport_target(cpp_ts_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(xbox_controller "${cpp_ts_target}")
target_link_libraries(robot_ctrl "${cpp_ts_target}")

ament_target_dependencies(xbox_controller
  rclcpp
  sensor_msgs
  std_msgs
)

ament_target_dependencies(robot_ctrl
  rclcpp
  sensor_msgs
  std_msgs
  ament_index_cpp
)

target_link_libraries(robot_ctrl xbox_controller ik_solver gait)
target_link_options(robot_ctrl PRIVATE "-Wl,-rpath,\$ORIGIN")

install(TARGETS xbox_controller ik_solver robot_ctrl
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS scripts/mujoco_simulator.py scripts/train_rl.py scripts/rl_inference.py
  scripts/leg_ik.py scripts/cpg_gait.py scripts/height_map.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rl_checkpoints rl_logs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

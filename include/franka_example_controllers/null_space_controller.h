// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_example_controllers/null_space_controller_paramConfig.h>
#include <franka_example_controllers/paramForDebug.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace franka_example_controllers {

class NullSpaceImpedanceController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface>
 {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated,const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;  //机器人全部状态
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;  //机器人的动力学和运动学模型
  std::vector<hardware_interface::JointHandle> joint_handles_;  //关节状态类

  double filter_params{0.005};//滤波参数，调整目标位置与阻抗变化速率

  const double delta_tau_max{1.0};//最大力矩变化值

  bool firstUpdate = true; //用于判断是不是第一个控制周期，计算雅可比导数。

  //主任务
  Eigen::Matrix<double, 3, 3> task1_Kp;
  Eigen::Matrix<double, 3, 3> task1_Kv;
  Eigen::Matrix<double, 3, 3> task1_Kp_target;
  Eigen::Matrix<double, 3, 3> task1_Kv_target;
  Eigen::Vector3d position_d;
  Eigen::Vector3d orientation_d;
  Eigen::Vector3d position_d_target;
  Eigen::Vector3d orientation_d_target;

  // 零空间任务 
  Eigen::Matrix<double, 7, 7> task2_Md;
  Eigen::Matrix<double, 7, 7> task2_Bd;
  Eigen::Matrix<double, 7, 7> task2_Kd;
  Eigen::Matrix<double, 7, 1> task2_q_d;
  Eigen::Matrix<double, 7, 1> task2_dq_d;
  Eigen::Matrix<double, 7, 1> task2_ddq_d;

  // 用于数值微分
  std::array<double, 42> jacobian_array_old;
  Eigen::Matrix<double, 3, 1> error_old;

  // 动态配置参数
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::null_space_controller_paramConfig>>dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::null_space_controller_paramConfig& config,uint32_t level);

  // 参数更新函数
  void controllerParamRenew();

  //求解欧拉角
  Eigen::Vector3d toEulerAngle(Eigen::Matrix3d R);

  ros::Publisher paramForDebug;
  
};

}  // namespace franka_example_controllers

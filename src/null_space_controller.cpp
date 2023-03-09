// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/null_space_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

#include <fstream>
#include <iostream>

namespace franka_example_controllers {

bool NullSpaceImpedanceController::init(hardware_interface::RobotHW* robot_hw,ros::NodeHandle& node_handle) 
{
  // 订阅话题：平衡点位置
  paramForDebug = node_handle.advertise<franka_example_controllers::paramForDebug>("paramForDebug",20);

  //参数服务器
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("NullSpaceImpedanceController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "NullSpaceImpedanceController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  //运动学/动力学模型类：实例化
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM("NullSpaceImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "NullSpaceImpedanceController: Exception getting model handle from interface: "<< ex.what());
    return false;
  }

  //机器人完整状态类：实例化
  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM("NullSpaceImpedanceController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("NullSpaceImpedanceController: Exception getting state handle from interface: "<< ex.what());
    return false;
  }
  
  //关节控制类（ROS自带）：实例化
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM("NullSpaceImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM("NullSpaceImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ = ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");
  dynamic_server_compliance_param_ = std::make_unique<dynamic_reconfigure::Server<franka_example_controllers::null_space_controller_paramConfig>>(dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(boost::bind(&NullSpaceImpedanceController::complianceParamCallback, this, _1, _2));

  //初始位置姿态赋初值
  position_d.setZero();
  orientation_d.setZero(); 
  position_d_target.setZero();
  orientation_d_target.setZero();

  // 控制参数赋初值
  task1_Kp.setIdentity();
  task1_Kv.setIdentity();
  task2_Md.setIdentity();
  task2_Bd.setIdentity();
  task2_Kd.setIdentity();

  return true;
}

// debug
int time = 0;
std::ofstream myfile;

void NullSpaceImpedanceController::starting(const ros::Time& /*time*/) 
{
  std::cout << "--------------start:NullSpaceImpedanceController_2.27--------------"<< std::endl;
  // 获取机器人初始状态
  franka::RobotState initial_state = state_handle_->getRobotState();
  // 基坐标系下的雅可比（上一时刻，用于数值微分）
  std::array<double, 42> jacobian_array_old = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array_old.data());
  std::cout << jacobian << std::endl;
  // 获取当前关节位置
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  // 当前笛卡尔位置的齐次变换矩阵
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // 将当前状态设置为平衡点
  position_d = initial_transform.translation();
  orientation_d = toEulerAngle(initial_transform.rotation());
  position_d_target = initial_transform.translation();
  orientation_d_target = toEulerAngle(initial_transform.rotation());

  // 零空间期望位置设为当其位置
  task2_q_d = q_initial;
  task2_dq_d.setZero();
  task2_ddq_d.setZero();
}

void NullSpaceImpedanceController::update(const ros::Time& /*time*/,const ros::Duration& t) 
{
  franka_example_controllers::paramForDebug param_debug;

  // 获取状态,C,M，q,dq
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 49> mass_array = model_handle_->getMass();
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // 将array类转成矩阵
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> mass(mass_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));  // 齐次变换矩阵
  Eigen::Vector3d position(transform.translation());
  Eigen::Vector3d orientation = toEulerAngle(transform.rotation());

  // 计算雅可比
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());//完整雅可比
  Eigen::Matrix<double, 3, 7> jacobian_1 = jacobian.block(0, 0, 3, 7); //位置雅可比
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_old(jacobian_array_old.data());  
  Eigen::Matrix<double, 3, 7> jacobian_1_old = jacobian_old.block(0, 0, 3, 7);
  Eigen::Matrix<double, 3, 7> jacobian_1_dot;
  if (firstUpdate)
  {
    // debug
    myfile.open("/home/wd/example_null.txt"); 
    myfile << "Writing this to a file.\n" << std::endl;

    jacobian_1_dot.setZero();  // 若在第一次控制周期，将雅可比导数置为0
  }
  else
  {
    jacobian_1_dot = (jacobian_1 - jacobian_1_old) / t.toSec();
    jacobian_array_old = jacobian_array;
  }

  // 误差计算
  // pos + ori
  /*   Eigen::Matrix<double, 6, 1> error;
    Eigen::Matrix<double, 3, 1> error_orientation;
    Eigen::Matrix<double, 6, 1> error_dot;
    error.head(3) << position_d - position;
    for(int i = 0; i < 3; i++)
    {
      if (std::fabs(orientation_d(i) - orientation(i)) <= M_PI)
      {
        error_orientation(i) = orientation_d(i) - orientation(i);
      }
      else
      {
        if (orientation(i) <= 0)
        {
          error_orientation(i) = 2 * M_PI - std::fabs(orientation_d(i) - orientation(i));
        }
        else
        {
          error_orientation(i) = - (2 * M_PI - std::fabs(orientation_d(i) - orientation(i)));
        }
      }
    }
    error.tail(3) << error_orientation;
    if (firstUpdate)
    {
      error_dot.setZero();
      firstUpdate = false;
    }
    else
    {
      error_dot = (error - error_old) / t.toSec();
      error_old = error;
    } */
  // pos
  Eigen::Matrix<double, 3, 1> error;
  Eigen::Matrix<double, 3, 1> error_dot;
  error << position_d - position;
  if (firstUpdate) 
  {
    error_dot.setZero();
    firstUpdate = false;
  } 
  else 
  {
    error_dot = (error - error_old) / t.toSec();
    error_old = error;
  }
  
  // 伪逆矩阵
  Eigen::MatrixXd jacobian_1_pinv;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
  weightedPseudoInverse(jacobian_1, jacobian_1_pinv, mass);

  // 命令加速度与输入力矩
  Eigen::VectorXd qc1(7), qc2(7), tau_d(7);
  qc1 << jacobian_1_pinv * (/*ddq + */ task1_Kp * error + task1_Kv * error_dot - jacobian_1_dot * dq);
  Eigen::MatrixXd N = I - jacobian_1_pinv * jacobian_1;
  qc2 << N * (/*task2_ddq_d = 0*/ mass.inverse() *(task2_Kd * (task2_q_d - q) + task2_Bd * (task2_dq_d - dq)));
  tau_d << mass * (qc1 + qc2) + coriolis;

  // debug
  time++;
  myfile << " " << std::endl;
  myfile << "time: " << time << std::endl;
  myfile << "task1_Kp: " << std::endl;
  myfile << task1_Kp << std::endl;
  myfile << "task1_Kv: " << std::endl;
  myfile << task1_Kv << std::endl;
  myfile << "R: " << std::endl;
  myfile << transform.rotation() << std::endl;
  myfile << "" << std::endl;
  myfile << "q: " << q.transpose() << std::endl;
  myfile << "dq: " << dq.transpose() << std::endl;
  myfile << "task2_q_d: " << task2_q_d.transpose() << std::endl;
  myfile << "task2_dq_d: " << task2_dq_d.transpose() << std::endl;
  myfile << "task2_ddq_d: " << task2_ddq_d.transpose() << std::endl;
  myfile << "position_d: " << position_d.transpose() << std::endl;
  myfile << "position: " << position.transpose() << std::endl;
  myfile << "orientation_d: " << orientation_d.transpose() << std::endl;
  myfile << "orientation: " << orientation.transpose() << std::endl;
  myfile << "error: " << error.transpose() << std::endl;
  myfile << "error_dot: " << error_dot.transpose() << std::endl;
  myfile << "jacobian:" << std::endl;
  myfile << jacobian << std::endl;
  myfile << "jacobian_1:" << std::endl;
  myfile << jacobian_1 << std::endl;
  myfile << "jacobian_1_dot:" << std::endl;
  myfile << jacobian_1_dot << std::endl;
  myfile << "jacobian_1_pinv:" << std::endl;
  myfile << jacobian_1_pinv << std::endl;
  Eigen::MatrixXd II = jacobian_1 * jacobian_1_pinv;
  myfile << "jacobian_1 * jacobian_1_pinv:" << std::endl;
  myfile << II << std::endl;
  myfile << "qc1: " << qc1.transpose() << std::endl;
  myfile << "qc2: " << qc2.transpose() << std::endl;
  myfile << "tau_d: " << tau_d.transpose() << std::endl;

  if ((std::fabs(tau_d(0)) > 1) || (std::fabs(tau_d(1)) > 1) || (std::fabs(tau_d(2)) > 1) ||
      (std::fabs(tau_d(3)) > 1) || (std::fabs(tau_d(4)) > 1) || (std::fabs(tau_d(5)) > 1) ||
      (std::fabs(tau_d(6)) > 1)) 
      {
/*     myfile << "time: " << time << std::endl;
    myfile << task1_Kp << std::endl;
    myfile << "task1_Kv: " << std::endl;
    myfile << task1_Kv << std::endl;
    myfile << "R: " << std::endl;
    myfile << transform.rotation() << std::endl;
    myfile << "" << std::endl;
    myfile << "q: " << q.transpose() << std::endl;
    myfile << "dq: " << dq.transpose() << std::endl;
    myfile << "task2_q_d: " << task2_q_d.transpose() << std::endl;
    myfile << "task2_dq_d: " << task2_dq_d.transpose() << std::endl;
    myfile << "task2_ddq_d: " << task2_ddq_d.transpose() << std::endl;
    myfile << "position_d: " << position_d.transpose() << std::endl;
    myfile << "position: " << position.transpose() << std::endl;
    myfile << "orientation_d: " << orientation_d.transpose() << std::endl;
    myfile << "orientation: " << orientation.transpose() << std::endl;
    myfile << "error: " << error.transpose() << std::endl;
    myfile << "error_dot: " << error_dot.transpose() << std::endl;
    myfile << "jacobian:" << std::endl;
    myfile << jacobian << std::endl;
    myfile << "jacobian_1:" << std::endl;
    myfile << jacobian_1 << std::endl;
    myfile << "jacobian_1_dot:" << std::endl;
    myfile << jacobian_1_dot << std::endl;
    myfile << "jacobian_1_pinv:" << std::endl;
    myfile << jacobian_1_pinv << std::endl;
    Eigen::MatrixXd II = jacobian_1 * jacobian_1_pinv;
    myfile << "jacobian_1 * jacobian_1_pinv:" << std::endl;
    myfile << II << std::endl;
    myfile << "qc1: " << qc1.transpose() << std::endl;
    myfile << "qc2: " << qc2.transpose() << std::endl;
    myfile << "tau_d: " << tau_d.transpose() << std::endl; */
  }
/*   printf("task1_Kp:  %f %f %f \n", task1_Kp(0,0), task1_Kp(1,1), task1_Kp(2,2));
  printf("task1_Kv:  %f %f %f \n", task1_Kv(0, 0), task1_Kv(1, 1), task1_Kv(2, 2));
  printf("task2_Kd:  %f %f %f \n", task2_Kd(0, 0), task2_Kd(1, 1), task2_Kd(2, 2));
  printf("task2_Bd:  %f %f %f \n", task2_Bd(0, 0), task2_Bd(1, 1), task2_Bd(2, 2));
  printf("task2_Md:  %f %f %f \n", task2_Md(0, 0), task2_Md(1, 1), task2_Md(2, 2));
  printf("position:  %f %f %f \n", position(0), position(1), position(2));
  printf("position_d:  %f %f %f \n", position_d(0), position_d(1), position_d(2));
  printf("position_error:  %f %f %f \n", error(0), error(1), error(2));
  printf("orientation:  %f %f %f \n", orientation(0), orientation(1), orientation(2));
  printf("orientation_d:  %f %f %f \n", orientation_d(0), orientation_d(1), orientation_d(2)); */
  printf("error:  %f %f %f \n", error(0), error(1), error(2));
  printf("error_dot:  %f %f %f \n", error_dot(0), error_dot(1), error_dot(2));
  printf("qc1:  %f %f %f %f %f %f %f\n", qc1(0), qc1(1), qc1(2), qc1(3), qc1(4), qc1(5), qc1(6));
  printf("qc2:  %f %f %f %f %f %f %f\n", qc2(0), qc2(1), qc2(2), qc2(3), qc2(4), qc2(5), qc2(6));
  printf("tau_d:  %f %f %f %f %f %f %f\n", tau_d(0), tau_d(1), tau_d(2), tau_d(3), tau_d(4),tau_d(5), tau_d(6));

  //画图
  for(int i = 0; i < 7; i++)
  {
    param_debug.qc1[i] = qc1[i];
    param_debug.qc2[i] = qc2[i];
    param_debug.tau_d[i] = tau_d[i];
  }
  paramForDebug.publish(param_debug);

  // 平滑命令
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));  // 关节句柄设置力矩命令
  }

  // 目标位置，控制参数更新
  controllerParamRenew();
}

Eigen::Matrix<double, 7, 1> NullSpaceImpedanceController::saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated,const Eigen::Matrix<double, 7, 1>& tau_J_d) 
{  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, delta_tau_max), - delta_tau_max);//6
  }
  return tau_d_saturated;
}

void NullSpaceImpedanceController::complianceParamCallback(franka_example_controllers::null_space_controller_paramConfig& config,uint32_t /*level*/) 
{
  task1_Kp_target.setIdentity();
  task1_Kp_target = config.task1_Kp * Eigen::Matrix3d::Identity();  // 左上3*3

  task1_Kv_target.setIdentity();
  task1_Kv_target = config.task1_Kv * Eigen::Matrix3d::Identity();

  task2_Kd = config.task2_Kd * Eigen::MatrixXd::Identity(7, 7);
  task2_Bd = config.task2_Bd * Eigen::MatrixXd::Identity(7, 7);
  task2_Md = config.task2_Md * Eigen::MatrixXd::Identity(7, 7);
}

void NullSpaceImpedanceController::controllerParamRenew() 
{
  task1_Kp = filter_params * task1_Kp_target + (1.0 - filter_params) * task1_Kp;
  task1_Kv = filter_params * task1_Kp_target + (1.0 - filter_params) * task1_Kv;

  task2_Kd = filter_params * task2_Kd + (1.0 - filter_params) * task2_Kd;
  task2_Bd = filter_params * task2_Bd + (1.0 - filter_params) * task2_Bd;
  task2_Md = filter_params * task2_Md + (1.0 - filter_params) * task2_Md;
}


//X-Y-Z固定角
Eigen::Vector3d NullSpaceImpedanceController::toEulerAngle(Eigen::Matrix3d R) 
{
  Eigen::Vector3d orientation;

  /*   for (int i = 0; i < 3; i++)  // 矩阵输入
    {
      for (int j = 0; j < 3; j++)
      {
        if (std::fabs(R(i, j)) < 0.0005)
        {
          R(i, j) = 0;
        }
      }
    } */

  //Y betha
  orientation(1) = atan2(-R(2, 0), std::fabs(std::sqrt(std::pow(R(0, 0), 2) + std::pow(R(1, 0), 2))));

  //Z alpha
  orientation(2) = atan2(R(1, 0) / cos(orientation(1)), R(0, 0) / cos(orientation(1)));

  //X r
  orientation(0) = atan2(R(2, 1) / cos(orientation(1)), R(2, 2) / cos(orientation(1)));

  return orientation;
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::NullSpaceImpedanceController,
                       controller_interface::ControllerBase)

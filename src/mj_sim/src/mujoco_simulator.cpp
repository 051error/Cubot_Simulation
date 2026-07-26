#include "mj_sim/mujoco_simulator.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <vector>

MujocoSimulator::MujocoSimulator()
  : Node("mujoco_simulator"), xbox_(this)
{
  RCLCPP_INFO(this->get_logger(), "Running Mujoco Simulator...");
  
  pos_puber_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/pos", 10);
  force_puber_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/force", 10);
  torque_puber_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/torque", 10);
  angle_puber_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/angle", 10);
  low_state_puber_ = this->create_publisher<mj_sim::msg::LowState>("/mujoco/low_state", 10);
  view_puber_ = this->create_publisher<sensor_msgs::msg::Image>("/mujoco/view", 10);
  last_ctrl_puber_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/last_ctrl", 10);
  low_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/mujoco/low_cmd", 10, [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        low_cmd_callback(msg);
      });
  this->init_mujoco();
  init_viewer();
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200), std::bind(&MujocoSimulator::publish_sensor_data, this));
  render_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(16), std::bind(&MujocoSimulator::render, this));
  sim_thread_ = std::thread(&MujocoSimulator::step_simulation, this);
}

MujocoSimulator::~MujocoSimulator()
{
  stop_simulation();
  close_viewer();
}

void MujocoSimulator::stop_simulation()
{
  if (sim_thread_.joinable()) {
    sim_thread_.join();
  }
}

void MujocoSimulator::init_mujoco()
{
  // Load URDF via MuJoCo 3.x spec API (native URDF support)
  std::string robot_path = ament_index_cpp::get_package_share_directory("cubot_description")
                         + "/cubot_urdf/urdf/cubot.urdf";
  char error[1000] = "Could not load URDF";

  mjSpec* spec = mj_parseXML(robot_path.c_str(), nullptr, error, 1000);
  if (!spec) {
    RCLCPP_ERROR(this->get_logger(), "Failed to parse robot URDF: %s", error);
    rclcpp::shutdown();
    return;
  }

  m_ = mj_compile(spec, nullptr);
  mj_deleteSpec(spec);

  if (!m_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to compile model");
    rclcpp::shutdown();
    return;
  }

  d_ = mj_makeData(m_);
  if (!d_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to create data structure");
    mj_deleteModel(m_);
    rclcpp::shutdown();
    return;
  }

}

void MujocoSimulator::init_viewer()
{
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  if (!glfwInit()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to init GLFW (no GPU or display?)");
    return;
  }
  window_ = glfwCreateWindow(1200, 900, "MuJoCo", nullptr, nullptr);
  if (!window_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to create window (try: export MESA_GL_VERSION_OVERRIDE=3.3)");
    return;
  }
  glfwMakeContextCurrent(window_);
  glfwShowWindow(window_);
  glfwSwapInterval(1);  // enable vsync
  const char* gl_ver = (const char*)glGetString(GL_VERSION);
  RCLCPP_INFO(this->get_logger(), "OpenGL: %s", gl_ver ? gl_ver : "NULL");

  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultScene(&scn_);
  mjv_defaultPerturb(&pert_);
  mjr_defaultContext(&con_);
  mjv_makeScene(m_, &scn_, 2000);
  mjr_makeContext(m_, &con_, mjFONTSCALE_150);

  cam_.lookat[0] = 0;
  cam_.lookat[1] = 0;
  cam_.lookat[2] = 0.15;
  cam_.distance  = 1.2;
  cam_.elevation = -25;
  cam_.azimuth   = 135;
  opt_.flags[mjVIS_LIGHT] = 1;
  RCLCPP_INFO(this->get_logger(),
      "Camera: lookat=(%.2f,%.2f,%.2f) dist=%.2f elev=%.0f azim=%.0f",
      cam_.lookat[0], cam_.lookat[1], cam_.lookat[2],
      cam_.distance, cam_.elevation, cam_.azimuth);

  // Register mouse callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double y) {
    auto* s = static_cast<MujocoSimulator*>(glfwGetWindowUserPointer(w));
    mjv_moveCamera(s->getModel(), mjMOUSE_ZOOM, 0, -y*0.05, &s->getScene(), &s->getCamera());
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int btn, int act, int) {
    auto* s = static_cast<MujocoSimulator*>(glfwGetWindowUserPointer(w));
    s->setButton(btn, act);
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
    auto* s = static_cast<MujocoSimulator*>(glfwGetWindowUserPointer(w));
    s->mouseMove(x, y);
  });

  // Show all visual overlays
  opt_.flags[mjVIS_JOINT]  = 1;
  opt_.flags[mjVIS_ACTUATOR] = 1;
  opt_.frame = mjFRAME_BODY;

  viewer_running_ = true;
  RCLCPP_INFO(this->get_logger(),
      "Viewer init. bodies=%d, geoms=%d, nq=%d, nu=%d",
      m_->nbody, m_->ngeom, m_->nq, m_->nu);
  for (int i = 0; i < m_->ngeom && i < 5; i++)
    RCLCPP_INFO(this->get_logger(), "  geom[%d]: body=%d type=%d size=(%.3f,%.3f,%.3f)",
      i, m_->geom_bodyid[i], m_->geom_type[i],
      m_->geom_size[3*i], m_->geom_size[3*i+1], m_->geom_size[3*i+2]);
}

void MujocoSimulator::close_viewer()
{
  viewer_running_ = false;
  mjv_freeScene(&scn_);
  mjr_freeContext(&con_);
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();
  RCLCPP_INFO(this->get_logger(), "Viewer closed.");
}

void MujocoSimulator::step_simulation()
{
  while (this->viewer_running_ && rclcpp::ok()) {
    if (receive_data) {
      mj_step(m_, d_);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void MujocoSimulator::render()
{
  static int frame_count = 0;
  if (!viewer_running_ || !window_ || !m_ || !d_) {
    if (frame_count == 0) RCLCPP_ERROR(this->get_logger(), "render: early return! vr=%d win=%p m=%p d=%p",
        viewer_running_, (void*)window_, (void*)m_, (void*)d_);
    return;
  }

  glfwMakeContextCurrent(window_);
  glfwPollEvents();
  if (glfwWindowShouldClose(window_)) {
    viewer_running_ = false;
    return;
  }

  int w, h;
  glfwGetFramebufferSize(window_, &w, &h);
  mjrRect viewport = {0, 0, w, h};

  // Render MuJoCo scene
  mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
  mjr_render(viewport, &scn_, &con_);

  // Read pixels BEFORE swap (back buffer = what we just rendered)
  img_buf_.resize(w * h * 3);
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img_buf_.data());
  auto img = sensor_msgs::msg::Image();
  img.header.stamp = this->now();
  img.header.frame_id = "world";
  img.height = h; img.width = w;
  img.encoding = "rgb8";
  img.is_bigendian = false;
  img.step = w * 3;
  img.data = img_buf_;
  view_puber_->publish(img);
  glfwSwapBuffers(window_);
  if (++frame_count <= 2)
    RCLCPP_INFO(this->get_logger(), "View published %dx%d to /mujoco/view", w, h);
}

void MujocoSimulator::publish_sensor_data()
{
  if (!m_ || !d_) return;

  auto msg = mj_sim::msg::LowState();

  // Leg joints: 6 legs x 3 DOF = 18
  for (int i = 0; i < 18 && i < m_->nq; i++) {
    msg.leg_pos[i] = d_->qpos[i];
    msg.leg_vel[i] = d_->qvel[i];
    msg.leg_torque[i] = d_->qfrc_actuator[i];
  }

  // Wheel joints after leg DOFs
  int ws = 18;
  for (int i = 0; i < 4 && (ws + i) < m_->nq; i++) {
    msg.wheel_pos[i] = d_->qpos[ws + i];
    msg.wheel_vel[i] = d_->qvel[ws + i];
  }

  // IMU from MP_BODY frame
  int body_id = mj_name2id(m_, mjOBJ_BODY, "MP_BODY");
  if (body_id >= 0) {
    int qidx = body_id * 4;
    msg.imu_quat[0] = d_->xquat[qidx + 0];
    msg.imu_quat[1] = d_->xquat[qidx + 1];
    msg.imu_quat[2] = d_->xquat[qidx + 2];
    msg.imu_quat[3] = d_->xquat[qidx + 3];
    msg.imu_gyro[0] = d_->qvel[body_id * 6 + 3];
    msg.imu_gyro[1] = d_->qvel[body_id * 6 + 4];
    msg.imu_gyro[2] = d_->qvel[body_id * 6 + 5];
  }

  low_state_puber_->publish(msg);
}

void MujocoSimulator::get_control_inputs(double* ctrl, int n) {
    for (int i = 0; i < n; i++)
        ctrl[i] = 0.0;  // TODO
}

void MujocoSimulator::low_cmd_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() != m_->nu) {
        return;
    }
    for (int i = 0; i < m_->nu; ++i) {
        d_->ctrl[i] = msg->data[i];
    }
    receive_data = true;
}

void MujocoSimulator::setButton(int btn, int act) {
  button_[btn] = (act == GLFW_PRESS);
  glfwGetCursorPos(window_, &last_mx_, &last_my_);
}

void MujocoSimulator::mouseMove(double x, double y) {
  if (!button_[GLFW_MOUSE_BUTTON_LEFT] &&
      !button_[GLFW_MOUSE_BUTTON_MIDDLE] &&
      !button_[GLFW_MOUSE_BUTTON_RIGHT]) return;
  double dx = x - last_mx_;
  double dy = y - last_my_;
  last_mx_ = x;
  last_my_ = y;
  int act = button_[GLFW_MOUSE_BUTTON_LEFT]  ? mjMOUSE_ROTATE_V :
            button_[GLFW_MOUSE_BUTTON_MIDDLE] ? mjMOUSE_ZOOM :
            mjMOUSE_MOVE_V;
  mjv_moveCamera(m_, act, dx*0.005, dy*0.005, &scn_, &cam_);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MujocoSimulator>();

    // Blocking render loop on main thread (GL context stays here)
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <thread>
#include "mj_sim/low_level_ctrl.hpp"
#include "mj_sim/msg/low_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <vector>


class MujocoSimulator : public rclcpp::Node
{
public:
    MujocoSimulator();
    ~MujocoSimulator();

    // For GLFW mouse callbacks
    mjModel*    getModel()  { return m_; }
    mjvScene&   getScene()  { return scn_; }
    mjvCamera&  getCamera() { return cam_; }
    void setButton(int btn, int act);
    void mouseMove(double x, double y);
    void render();
    bool viewer_running_ = false;

private:
    void init_mujoco();
    void init_viewer();
    void close_viewer();
    void step_simulation();
    void stop_simulation();
    void publish_sensor_data();
    void get_control_inputs(double* ctrl, int n);
    void low_cmd_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr render_timer_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pos_puber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr force_puber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_puber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr angle_puber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr last_ctrl_puber_;
    rclcpp::Publisher<mj_sim::msg::LowState>::SharedPtr low_state_puber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr view_puber_;
    std::vector<uint8_t> img_buf_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_pos_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_angle_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr low_cmd_sub_;
    std::thread sim_thread_;
    XboxController xbox_;

    mjModel* m_ = nullptr;
    mjData* d_ = nullptr;

    // MuJoCo passive viewer state
    mjvCamera   cam_  = {};
    mjvOption   opt_  = {};
    mjvScene    scn_  = {};
    mjvPerturb  pert_ = {};
    mjrContext  con_  = {};
    GLFWwindow* window_ = nullptr;
    // Mouse state for camera control
    bool button_[8] = {};
    double last_mx_ = 0, last_my_ = 0;
    bool receive_data = false;
};


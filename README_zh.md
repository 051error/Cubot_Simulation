# cubot_ws

[English](README.md)

**cubot** 的 ROS 2 + MuJoCo 仿真与控制栈。cubot 是一台六足机器人，采用
PhantomX 式三自由度腿部布局（偏航 coxa + 俯仰 femur + 俯仰 tibia）。运动由
中枢模式发生器（CPG）生成，并通过逆运动学（IK）求解；同时提供一版在 CPG
层之上训练的强化学习（RL）策略。

## 概述

整个栈围绕一条三阶段管线组织：

```
CPG（Hopf 振荡器）  →  足端目标  →  IK  →  关节角度  →  MuJoCo
```

- **三足步态 CPG** 为每条腿产生相位信号（`x`）与正交信号（`y`）。
- **足端轨迹** 将这些信号映射为机体坐标系下的足端目标，目标沿与指令速度
  相反的方向扫过一条直线。
- **解析 IK**（C++）或 **KD-tree IK**（Python，供 RL 使用）把每个足端目标
  转换为三个绝对关节角度。
- **MuJoCo 仿真器** 积分物理并把机器人状态以 200 Hz 回传给控制器。

## 特性

- **三种运动模式**，用手柄肩键切换：
  - **NORMAL** — 摇杆驱动的 CPG 行走（前进 / 后退 / 侧向）。
  - **TURN** — 固定参数的 CPG 原地旋转。
  - **RL** — 训练好的 PPO 策略通过 CPG 层驱动全部 18 个腿部关节。
- **机体坐标系下的直线足端扫掠**（无 coxa 圆弧），消除了圆弧轨迹在边角腿
  上泄漏出来的侧向打滑。
- **两套 IK 后端**共享同一套腿部几何：用于实时控制的闭式 C++ 求解器，以及
  用于 RL 训练的快速 KD-tree 查表。
- 自动化的**打滑 / 直线**与**转向平衡**测试脚本。

## 包

| 包 | 描述 |
| --- | --- |
| `cubot_description` | URDF/xacro 机器人描述、网格、RViz 配置。 |
| `mj_sim` | MuJoCo 仿真器、控制器、CPG/IK/RL，以及测试脚本。 |

## 架构

`mj_sim` 拆分为纯 C++ 的**步态**模块和 ROS 感知的**控制器**：

```
mj_sim/
├── include/mj_sim/
│   ├── gait.hpp          # GaitGenerator、TripodGait、FootTrajectory、TurnGait、LegGeometry
│   ├── ik_solver.hpp     # LegIK（三自由度解析逆运动学）
│   └── robot_ctrl.hpp    # XboxController + RobotController
├── src/
│   ├── gait.cpp
│   ├── ik_solver.cpp
│   ├── robot_ctrl.cpp    # 模式状态机 + ROS 收发（main）
│   └── xbox_controller.cpp
├── msg/
│   ├── LowState.msg      # /mujoco/low_state（关节、IMU、足端、接触）
│   └── UpCmd.msg         # /upper_ctrl（高层速度指令）
├── models/
│   ├── scene.xml         # 世界 + 地面
│   ├── terrain.xml       # 平地 + 台阶 + 起伏凸起（RL 地形）
│   ├── cubot.xml         # 机器人 MJCF（由 convert_urdf.py 生成）
│   └── meshes/           # 模型网格（与 cubot.xml 同目录，保证相对路径）
└── scripts/
    ├── mujoco_simulator.py            # MuJoCo 物理 + /mujoco/low_state 发布者
    ├── convert_urdf.py                # URDF → MJCF 转换器
    ├── leg_ik.py                      # KD-tree IK（RL 使用）
    ├── hexapod_cpg.py                 # Hopf 振荡器 CPG（RL 使用）
    ├── train_rl.py                    # PPO 训练（参考 arXiv:2310.07744）
    ├── rl_inference.py                # RL 策略节点
    ├── test_straight_line.py          # 打滑 + 直线测试
    ├── test_turn_balance.py           # 转向平衡测试
    └── analysis_workspace_velocity.py # 足端工作空间 / CPG 速度分析
```

## 依赖

- **Ubuntu**（在 24.04 上测试）搭配 **ROS 2 Jazzy**。
- **MuJoCo 3.3.2** 与 `mujoco-viewer`（pip）。
- ROS 包：`rclcpp`、`sensor_msgs`、`std_msgs`、`ament_index_cpp`、`joy`。
- Python（仿真 / RL 用）：`numpy`、`scipy`、`gymnasium`、`stable-baselines3`、
  `torch`。

## 构建

```bash
cd cubot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 运行

在两个终端中分别启动仿真器（会打开 MuJoCo 视图）和控制器：

```bash
# 终端 1 —— 物理 + 状态发布者
ros2 run mj_sim mujoco_simulator.py

# 终端 2 —— 控制器（同时 fork 出 joy_node 以支持手柄）
ros2 run mj_sim robot_ctrl
```

视图需要图形界面；通过 SSH 运行时请设置 `DISPLAY`。

## 控制模式

手柄映射（Xbox 布局）：

| 输入 | 含义 |
| --- | --- |
| 左摇杆 Y（`axes[1]`） | 前进 / 后退（NORMAL） |
| 左摇杆 X（`axes[0]`） | 侧向左 / 右（NORMAL） |
| 右摇杆 X（`axes[3]`） | 转向方向与速率（TURN） |
| `RB`（buttons[5]） | TURN 模式（按住） |
| `LB` + `RB` | RL 模式（按住） |
| `Y`（buttons[3]） | 打开盖板 |

- **NORMAL** — 不按肩键。左摇杆驱动 CPG 行走。
- **TURN** — 按住 `RB`，右摇杆左右推进行原地旋转。
- **RL** — 同时按住 `LB` + `RB`，并运行 `rl_inference.py`：

  ```bash
  ros2 run mj_sim rl_inference.py
  ```

## ROS 2 话题

| 话题 | 类型 | 方向 | 用途 |
| --- | --- | --- | --- |
| `/mujoco/low_cmd` | `Float64MultiArray` | 控制器 → 仿真 | 19 个关节指令（18 腿 + 盖板） |
| `/mujoco/low_state` | `mj_sim/LowState` | 仿真 → 控制器 | 关节位置/速度/力矩、IMU、足端、接触 |
| `/mujoco/last_ctrl` | `Float64MultiArray` | 仿真 → 调试 | 最近一次施加的控制量 |
| `/joy` | `sensor_msgs/Joy` | 手柄 → 控制器 | 原始摇杆数据 |
| `/upper_ctrl` | `mj_sim/UpCmd` | 控制器 → RL | 高层速度指令 |
| `/rl_action` | `Float64MultiArray` | RL → 控制器 | 策略驱动的 18 个关节目标 |

## 测试

两个自动化脚本用于测量控制器关心的行为。它们会自行发布 `/joy`，因此请先
停掉 `joy_node`（或实体手柄），避免两个发布者争抢同一话题。在仿真器已启动、
且已 source 工作区（以便解析 `mj_sim.msg` 定义）的前提下运行：

```bash
# 打滑 + 直线测试（NORMAL 模式，先前进后后退）
python3 src/mj_sim/scripts/test_straight_line.py --duration 15

# 转向平衡测试（TURN 模式，先快后慢）
python3 src/mj_sim/scripts/test_turn_balance.py --duration 30
```

## RL 训练

RL 管线沿用 [arXiv:2310.07744](https://arxiv.org/abs/2310.07744)（面向六足
运动的、具备地形自适应能力的 CPG + RL）的架构：策略输出 8 个 CPG 足端轨迹
参数，Hopf 振荡器将其转化为足端位置，KD-tree IK 再解出关节角度。

通过 `--scene` 可选两种场景：

| 场景 | 地面 |
| --- | --- |
| `scene.xml`（默认） | 平坦地面 |
| `terrain.xml` | 平地 + 台阶 + 平台 + 起伏凸起 |

```bash
# 在平坦地面上训练（默认）
python3 src/mj_sim/scripts/train_rl.py --total_steps 5000000

# 在起伏地形上训练
python3 src/mj_sim/scripts/train_rl.py --scene terrain.xml --total_steps 5000000

# 运行训练好的策略（发布 /rl_action）
ros2 run mj_sim rl_inference.py
```

## 坐标系约定

- **机体坐标系**：X = 前方，Y = 左方，Z = 上方。
- **腿序**：`rf, rm, rr, lf, lm, lr`（右前 … 左后）。
- **关节**：角度 `0` = 腿伸直。蹲伏参考姿态为 `coxa = 0`、`thigh = -0.7593`、
  `tibia = -0.7108`（六条腿一致）。
- **腿部坐标系（c1_rest）**：X = 竖直（向上），Y = 前方/向外，Z = 侧向。

## 致谢

本项目的底盘与六足模型源自开源项目
[phantomx_description](https://github.com/HumaRobotics/phantomx_description)（HumaRobotics）。

## 许可证

Apache-2.0（见 `src/mj_sim/LICENSE` 与 `src/cubot_description/LICENSE`）。

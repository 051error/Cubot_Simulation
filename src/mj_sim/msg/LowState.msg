# LowState - combined leg joint state + IMU data
float64[18] leg_pos       # 6 legs x 3 joints (coxa, femur, tibia)
float64[18] leg_vel       # joint velocities
float64[18] leg_torque    # joint torques
float64[4]  imu_quat      # body orientation quaternion (w, x, y, z)
float64[3]  imu_gyro      # body angular velocity (world frame, x, y, z)

# Task-space observation fields for RL policy
float64[3]  body_vel      # body-frame velocity (vx, vy, wz)
float64[18] feet_pos      # 6 feet x (x,y,z) in body frame
float64[18] feet_vel      # 6 feet x (vx,vy,vz) in body frame
float64[6]  foot_contact  # ground contact: 1=touching, 0=in air

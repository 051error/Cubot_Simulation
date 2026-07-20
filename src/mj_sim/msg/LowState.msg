# LowState - combined leg/wheel joint state + IMU data
float64[18] leg_pos       # 6 legs x 3 joints (coxa, femur, tibia)
float64[18] leg_vel       # joint velocities
float64[18] leg_torque    # joint torques
float64[4]  wheel_pos     # 4 drive wheels
float64[4]  wheel_vel     # wheel velocities
float64[4]  imu_quat            # orientation quaternion (w, x, y, z)
float64[3]  imu_gyro            # angular velocity (x, y, z)

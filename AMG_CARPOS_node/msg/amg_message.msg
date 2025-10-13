string space #Options: 'task', 'joint'
string motion_type #Options: 'reach', 'poly', 'home'
geometry_msgs/Pose target_pose 
float64[6] target_config
float64 duration # in seconds
string end_effector #Options: 'camera', 'gripper'
float64 tomato_r #THIS IS ADDED FOR THE SCALING

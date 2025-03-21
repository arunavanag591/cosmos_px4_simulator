# OdorReading.msg
# A simple custom message to store odor concentration at a given position.

# Standard ROS Header provides timestamp and frame_id
std_msgs/Header header

# Odor concentration measured (arbitrary units)
float32 concentration

# Where this odor measurement was taken
geometry_msgs/Point position

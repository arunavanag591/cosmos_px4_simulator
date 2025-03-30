#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, TwistStamped, Vector3
import numpy as np
from std_msgs.msg import Float32

class WindEstimator:
    def __init__(self):
        rospy.init_node('wind_estimator')
        
        # Buffer size for moving average
        self.buffer_size = 20
        
        # Initialize buffers
        self.wind_buffer_x = []
        self.wind_buffer_y = []
        self.wind_buffer_z = []
        
        # Publishers and subscribers
        rospy.Subscriber('/mavros/setpoint_velocity/cmd_vel', Twist, self.cmd_vel_cb)
        rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped, self.actual_vel_cb)
        self.wind_pub = rospy.Publisher('/estimated_wind', Vector3, queue_size=10)
        self.wind_magnitude_pub = rospy.Publisher('/wind_magnitude', Float32, queue_size=10)
        
        # Initialize velocity variables
        self.cmd_vel = Twist()
        self.actual_vel = TwistStamped().twist
        self.has_cmd_vel = False
        self.has_actual_vel = False
        
        # Timer for publishing at fixed rate
        rospy.Timer(rospy.Duration(0.1), self.estimate_wind)
        
        rospy.loginfo("Wind estimator initialized")
    
    def cmd_vel_cb(self, msg):
        self.cmd_vel = msg
        self.has_cmd_vel = True
    
    def actual_vel_cb(self, msg):
        self.actual_vel = msg.twist
        self.has_actual_vel = True
    
    def estimate_wind(self, event):
        if not (self.has_cmd_vel and self.has_actual_vel):
            return
        
        # Calculate wind as difference between actual and commanded velocity
        wind_x = self.actual_vel.linear.x - self.cmd_vel.linear.x
        wind_y = self.actual_vel.linear.y - self.cmd_vel.linear.y
        wind_z = self.actual_vel.linear.z - self.cmd_vel.linear.z
        
        # Add to buffers
        self.wind_buffer_x.append(wind_x)
        self.wind_buffer_y.append(wind_y)
        self.wind_buffer_z.append(wind_z)
        
        # Maintain buffer size
        if len(self.wind_buffer_x) > self.buffer_size:
            self.wind_buffer_x.pop(0)
            self.wind_buffer_y.pop(0)
            self.wind_buffer_z.pop(0)
        
        # Calculate filtered values using moving average
        avg_wind_x = sum(self.wind_buffer_x) / len(self.wind_buffer_x)
        avg_wind_y = sum(self.wind_buffer_y) / len(self.wind_buffer_y)
        avg_wind_z = sum(self.wind_buffer_z) / len(self.wind_buffer_z)
        
        # Calculate wind magnitude
        wind_magnitude = np.sqrt(avg_wind_x**2 + avg_wind_y**2 + avg_wind_z**2)
        
        # Publish vector and magnitude
        wind_vec = Vector3()
        wind_vec.x = avg_wind_x
        wind_vec.y = avg_wind_y
        wind_vec.z = avg_wind_z
        self.wind_pub.publish(wind_vec)
        
        wind_mag = Float32()
        wind_mag.data = wind_magnitude
        self.wind_magnitude_pub.publish(wind_mag)
        
        # Log for debugging
        if rospy.get_time() % 5 < 0.1:  # Log approximately every 5 seconds
            rospy.loginfo("Wind estimate - X: %.2f, Y: %.2f, Z: %.2f, Magnitude: %.2f", 
                         avg_wind_x, avg_wind_y, avg_wind_z, wind_magnitude)

if __name__ == '__main__':
    try:
        estimator = WindEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
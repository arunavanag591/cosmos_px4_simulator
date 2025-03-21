#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
import math

# Global variables
current_state = State()
current_pose = PoseStamped()

def state_cb(msg):
    global current_state
    current_state = msg

def pose_cb(msg):
    global current_pose
    current_pose = msg

def wait_for_fcu_connection():
    """Wait for the FCU connection"""
    rospy.loginfo("Waiting for FCU connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()
    rospy.loginfo("FCU connected!")

if __name__ == "__main__":
    rospy.init_node("offboard_controller")
    
    # Subscribers
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
    pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=pose_cb)
    
    # Publishers
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    
    # Services
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
    
    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
    
    rate = rospy.Rate(20)  # 20Hz
    
    # Wait for FCU connection
    wait_for_fcu_connection()
    
    # Send a few setpoints before starting
    setpoint = PoseStamped()
    setpoint.pose.position.x = 30.0  # Starting position
    setpoint.pose.position.y = 6.0
    setpoint.pose.position.z = 1.0
    setpoint.pose.orientation.w = 1.0
    
    # Send setpoints for 2 seconds before attempting to switch modes
    rospy.loginfo("Sending initial setpoints...")
    for i in range(40):  # 2 seconds at 20Hz
        local_pos_pub.publish(setpoint)
        rate.sleep()
    
    # Try to switch to OFFBOARD mode and arm
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'
    
    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True
    
    last_request = rospy.Time.now()
    
    # Main loop
    rospy.loginfo("Attempting to switch to OFFBOARD mode and arm...")
    while not rospy.is_shutdown():
        # Update timestamp
        setpoint.header.stamp = rospy.Time.now()
        local_pos_pub.publish(setpoint)
        
        # Check if we need to send an offboard request
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_request) > rospy.Duration(2.0):
            response = set_mode_client.call(offb_set_mode)
            if response.mode_sent:
                rospy.loginfo("OFFBOARD mode enabled")
            else:
                rospy.logwarn(f"Failed to set OFFBOARD mode. Current mode: {current_state.mode}")
            last_request = rospy.Time.now()
        
        # Check if we need to arm
        if not current_state.armed and (rospy.Time.now() - last_request) > rospy.Duration(2.0):
            response = arming_client.call(arm_cmd)
            if response.success:
                rospy.loginfo("Vehicle armed")
            else:
                rospy.logwarn("Failed to arm vehicle")
            last_request = rospy.Time.now()
        
        # Once armed and in OFFBOARD mode, we let the plume tracker control the drone
        if current_state.armed and current_state.mode == "OFFBOARD":
            # Log status every 5 seconds
            if int(rospy.Time.now().to_sec()) % 5 == 0:
                rospy.loginfo(f"Drone armed and in OFFBOARD mode. Plume tracker is controlling.")
        
        rate.sleep()
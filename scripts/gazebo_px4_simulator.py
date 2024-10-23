#! /usr/bin/env python

import rospy
import csv
import rospkg
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
import math
import time

current_state = State()
current_pose = PoseStamped()

# Current target point
tx, ty, tz, tyaw = 0, 0, 2, 0


def state_cb(msg):
    global current_state
    current_state = msg

def pose_cb(msg):
    global current_pose
    current_pose = msg

def shutdown_handler():
    rospy.loginfo("Shutting down offboard node.")

def wait_for_system_health():
    rospy.loginfo("Waiting for system health checks to pass...")
    while not rospy.is_shutdown():
        if current_state.system_status == 3:  # MAV_STATE_STANDBY (indicates preflight checks passed)
            rospy.loginfo("System health checks passed.")
            break
        rospy.logwarn("System not ready: Resolve system health failures before arming.")
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("gazebo_px4_simulator")
    rospy.on_shutdown(shutdown_handler)

    # Subscribers and Publishers
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
    pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=pose_cb)
    setpoint_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
    velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", Twist, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
    rate = rospy.Rate(100)

    # Wait for Flight Controller connection
    while not rospy.is_shutdown() and not current_state.connected:
        rospy.loginfo("Waiting for FCU connection...")
        rate.sleep()

    # Load trajectory from CSV
    trajectory = []
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('gazebo_px4_simulator')
        csv_path = f"{package_path}/trajectories/trajectory.csv"
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 4:
                    waypoint = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                    trajectory.append(waypoint)
                    rospy.loginfo(f"Loaded waypoint: {waypoint}")
    except FileNotFoundError:
        rospy.logerr("Trajectory CSV file not found!")
        exit(1)

    if not trajectory:
        rospy.logerr("No valid waypoints found in CSV file!")
        exit(1)

    # Initial takeoff setpoint
    takeoff_pose = PositionTarget()
    takeoff_pose.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
    takeoff_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ + \
                             PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                             PositionTarget.IGNORE_YAW_RATE
    takeoff_pose.position.x = 0
    takeoff_pose.position.y = 0
    takeoff_pose.position.z = 2  # Set an altitude of 2 meters for takeoff
    takeoff_pose.yaw = 0  # Set initial yaw to 0 radians

    # Send a few setpoints before starting
    for _ in range(400):  # Increased from 200 to 400 to ensure enough setpoints are sent
        if rospy.is_shutdown():
            break
        setpoint_pub.publish(takeoff_pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    # Wait for system health before attempting to arm
    wait_for_system_health()

    # Ensure OFFBOARD mode and arming before starting the mission
    while not rospy.is_shutdown() and (not current_state.armed or current_state.mode != "OFFBOARD"):
        setpoint_pub.publish(takeoff_pose)

        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
            response = set_mode_client.call(offb_set_mode)
            if response.mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            else:
                rospy.logwarn("Failed to set OFFBOARD mode. Retrying...")
            last_req = rospy.Time.now()

        if not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
            response = arming_client.call(arm_cmd)
            if response.success:
                rospy.loginfo("Vehicle armed")
            else:
                rospy.logwarn("Failed to arm vehicle. Retrying...")
            last_req = rospy.Time.now()

        rate.sleep()

    # Hold position at takeoff altitude until stable
    rospy.loginfo("Holding at takeoff altitude...")
    while not rospy.is_shutdown():
        current_position = current_pose.pose.position
        distance = math.sqrt((current_position.x - takeoff_pose.position.x) ** 2 +
                             (current_position.y - takeoff_pose.position.y) ** 2 +
                             (current_position.z - takeoff_pose.position.z) ** 2)
        if distance < 0.5:  # Check if the drone has reached the takeoff altitude within 0.5 meters
            rospy.loginfo("Takeoff position reached. Proceeding to waypoint navigation.")
            break
        setpoint_pub.publish(takeoff_pose)
        rate.sleep()

    # Start waypoint navigation (using austin's constant vel)
    waypoint_index = 0
    velocity_cmd = Twist()
    threshold_distance = 0.5  # Threshold to switch waypoints
    V = 0.5  # Velocity magnitude for movement

    while not rospy.is_shutdown() and waypoint_index < len(trajectory):
        # Set the target waypoint
        tx, ty, tz, tyaw = trajectory[waypoint_index]

        # Create PositionTarget message to combine position and yaw
        position_target = PositionTarget()
        position_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        position_target.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ + \
                                    PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
        position_target.position.x = tx
        position_target.position.y = ty
        position_target.position.z = tz
        position_target.yaw = tyaw

        # Calculate position error 
        ex = tx - current_pose.pose.position.x
        ey = ty - current_pose.pose.position.y
        ez = tz - current_pose.pose.position.z
        distance_to_target = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

        # Normalize error to get direction
        if distance_to_target > 0:
            en = np.array([ex, ey, ez]) / distance_to_target
        else:
            en = np.array([0, 0, 0])

        # Set velocity command in the direction of the target
        velocity_cmd.linear.x = V * en[0]
        velocity_cmd.linear.y = V * en[1]
        velocity_cmd.linear.z = V * en[2]

        # Publish position and velocity commands
        setpoint_pub.publish(position_target)
        velocity_pub.publish(velocity_cmd)

        # Check if the drone is close enough to the target to move to the next waypoint
        if distance_to_target < threshold_distance:
            rospy.loginfo(f"Reached waypoint {waypoint_index + 1} of {len(trajectory)}. Moving to next waypoint...")
            waypoint_index += 1


        rate.sleep()

    # Stop the drone after reaching all waypoints
    rospy.loginfo("All waypoints reached. Entering standby mode.")
    velocity_cmd.linear.x = 0
    velocity_cmd.linear.y = 0
    velocity_cmd.linear.z = 0
    while not rospy.is_shutdown():
        velocity_pub.publish(velocity_cmd)  # Keep publishing zero velocity to maintain stability
        rate.sleep()

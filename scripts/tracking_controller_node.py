#! /usr/bin/env python

import rospy
import csv
import rospkg
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from visualization_msgs.msg import Marker, MarkerArray
import math
import time
import tf.transformations

# For odor detection - assuming a similar implementation to the example
# You would need to create or import your odor simulator/predictor
try:
    from plume_sim_fast import CosmosFast
    import pandas as pd
except ImportError:
    rospy.logwarn("Could not import CosmosFast, will use dummy odor detection")

class OdorTrackerNode:
    def __init__(self):
        # Initialize node
        rospy.init_node("odor_tracking_controller")
        
        # State variables
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        
        # Flag to track if we've reached the starting position
        self.reached_start_position = False
        
        # Starting position (well inside the plume)
        self.start_x = 35.0
        self.start_y = 2.0
        
        # Odor detection variables
        self.current_odor = 0.0
        self.last_odor = 0.0
        self.odor_threshold = 4.5  # From example code
        self.hit_occurred = False
        self.last_hit_time = rospy.Time.now().to_sec()
        
        # Behavior parameters
        self.target_pos = np.array([0.0, 0.0])  # Odor source location (if known)
        self.closest_to_source = 0.5  # Distance to consider source reached
        self.is_surging = False
        self.is_casting = True  # Start with casting behavior
        self.surge_speed = 3.5  # m/s during surge
        self.base_speed = 1   # m/s during normal flight
        self.surge_duration = 2.0  # seconds to continue surge after losing odor
        self.surge_end_time = 0
        
        # Casting parameters
        self.cast_base_freq = 0.5   # Base frequency for sine wave (Hz)
        self.cast_growth_rate = 0.9 # Rate at which amplitude grows per second out of odor
        self.max_cast_amplitude = 5.0  # Maximum crosswind amplitude
        
        # Altitude to maintain
        self.altitude = 4.0  # meters
        
        # Load odor predictor if available
        self.load_odor_predictor()
        
        # ROS subscribers
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_callback)
        self.pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_callback)
        
        # ROS publishers
        self.setpoint_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        # Fix: Use TwistStamped instead of Twist for velocity commands
        from geometry_msgs.msg import TwistStamped
        self.velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
        self.drone_marker_pub = rospy.Publisher("drone_position_marker", Marker, queue_size=10)
        
        # Set up service clients
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        
        # Control rate (20Hz seems common for odor tracking)
        self.dt = 0.006  # 20 Hz
        self.rate = rospy.Rate(1.0 / self.dt)
        
        # Setup visualization markers
        self.setup_markers()
        
        rospy.loginfo("Odor tracking controller initialized")
    
    def load_odor_predictor(self):
        """Load odor prediction model if available"""
        self.predictor = None
        try:
            # Attempt to load the odor model - customize path as needed
            rospy.loginfo("Attempting to load odor model...")
            dirname = rospy.get_param('~odor_model_path',
                                  '/home/vbl/gazebo_ws/src/gazebo_px4_simulator/odor_sim_assets/hws/')
            hmap_data = np.load(str(dirname) + "hmap.npz")
            fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
            fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')
            
            self.predictor = CosmosFast(
                fitted_p_heatmap=hmap_data['fitted_heatmap'],
                xedges=hmap_data['xedges'],
                yedges=hmap_data['yedges'],
                fdf=fdf,
                fdf_nowhiff=fdf_nowhiff
            )
            rospy.loginfo("CosmosFast predictor initialized successfully")
        except Exception as e:
            rospy.logwarn(f"Could not load odor model: {e}")
            rospy.logwarn("Will use a simple simulated model instead")
            self.predictor = self.DummyPredictor()  # Fallback to a simple model
    
    class DummyPredictor:
        """Simple dummy odor predictor for testing"""
        def __init__(self):
            self.source_pos = np.array([0.0, 0.0])  # Odor source at origin
            self.plume_width = 3.0  # meters
            self.max_distance = 20.0  # How far odor can be detected
        
        def step_update(self, x, y, dt=0.05):
            """Return simulated odor concentration at position (x, y)"""
            # Calculate distance to source
            pos = np.array([x, y])
            distance = np.linalg.norm(pos - self.source_pos)
            
            # If we're too far, no odor
            if distance > self.max_distance:
                return 0.0
            
            # Calculate concentration based on distance and angle
            # This creates a downwind plume in +X direction
            if x > 0 and abs(y) < self.plume_width:
                # Exponential decay from source
                concentration = 10.0 * np.exp(-0.1 * distance)
                # Add some noise
                concentration *= (1.0 + 0.2 * np.random.randn())
                return max(0.0, concentration)
            else:
                return 0.0
    
    def setup_markers(self):
        """Setup visualization markers for RViz"""
        # Create marker array for visualization
        # self.markers = MarkerArray()
        
        # # Source marker (if known)
        # source_marker = Marker()
        # source_marker.header.frame_id = "map"
        # source_marker.header.stamp = rospy.Time.now()
        # source_marker.ns = "source"
        # source_marker.id = 0
        # source_marker.type = Marker.SPHERE
        # source_marker.action = Marker.ADD
        # source_marker.pose.position.x = self.target_pos[0]
        # source_marker.pose.position.y = self.target_pos[1]
        # source_marker.pose.position.z = 1.0
        # source_marker.pose.orientation.w = 1.0
        # source_marker.scale.x = 0.5
        # source_marker.scale.y = 0.5
        # source_marker.scale.z = 0.5
        # source_marker.color.r = 1.0
        # source_marker.color.g = 0.0
        # source_marker.color.b = 0.0
        # source_marker.color.a = 1.0
        # self.markers.markers.append(source_marker)
        
        # Start a marker publisher thread
        import threading
        marker_thread = threading.Thread(target=self.publish_markers)
        marker_thread.daemon = True
        marker_thread.start()
        
        # Start drone position marker thread
        drone_marker_thread = threading.Thread(target=self.publish_drone_marker)
        drone_marker_thread.daemon = True
        drone_marker_thread.start()
    
    def publish_markers(self):
        """Thread to publish markers"""
        while not rospy.is_shutdown():
            self.marker_pub.publish(self.markers)
            if hasattr(self, 'whiff_markers') and len(self.whiff_markers.markers) > 0:
                self.marker_pub.publish(self.whiff_markers)
            rospy.sleep(0.5)  # 2Hz is enough for markers
    
    def publish_drone_marker(self):
        """Thread to publish drone position marker"""
        while not rospy.is_shutdown():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "drone"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Set position to current drone position
            pos = self.current_position
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            
            # Default orientation
            marker.pose.orientation.w = 1.0
            
            # Set size and color
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            
            # Red for surging, blue for casting
            if self.is_surging:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            marker.color.a = 1.0
            
            self.drone_marker_pub.publish(marker)
            rospy.sleep(0.1)  # 10Hz for drone marker

    def add_whiff_marker(self, x, y, z):
        """Add a marker for an odor detection point"""
        # Create whiff markers array if it doesn't exist
        if not hasattr(self, 'whiff_markers'):
            self.whiff_markers = MarkerArray()
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "whiffs"
        marker.id = len(self.whiff_markers.markers)  # Increment ID for each marker
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        
        # Set size and color (red for whiff detection)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.whiff_markers.markers.append(marker)
        
        # Also publish this marker immediately
        self.marker_pub.publish(self.whiff_markers)
    
    def state_callback(self, msg):
        """Callback for drone state messages"""
        self.current_state = msg
    
    def pose_callback(self, msg):
        """Callback for drone position messages"""
        self.current_pose = msg
        
        # Extract position
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation_list)
        self.current_yaw = yaw
        
        # If in OFFBOARD mode and armed, run tracking algorithm
        if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
            self.tracking_step()
    
    # Flag to track if we've successfully reached the starting position
    reached_start_position = False
    
    def tracking_step(self):
        """Main tracking algorithm step"""
        # Skip tracking until we've confirmed reaching the starting position
        if not self.reached_start_position:
            # Check if we're at the starting position
            current_pos = self.current_position
            dist_to_start = math.sqrt(
                (current_pos[0] - self.start_x)**2 + 
                (current_pos[1] - self.start_y)**2
            )
            
            if dist_to_start < 3.0:  # Within 3m of starting position
                rospy.loginfo(f"Reached starting position. Beginning odor tracking.")
                self.reached_start_position = True
            else:
                # Still moving to starting position - send position commands only
                takeoff_pose = PositionTarget()
                takeoff_pose.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                takeoff_pose.type_mask = (
                    PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
                    PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
                    PositionTarget.IGNORE_YAW_RATE
                )
                takeoff_pose.position.x = self.start_x
                takeoff_pose.position.y = self.start_y
                takeoff_pose.position.z = self.altitude
                takeoff_pose.yaw = math.pi
                self.setpoint_pub.publish(takeoff_pose)
                return
        
        current_time = rospy.Time.now().to_sec()
        
        # Get current position
        x, y = self.current_position[0], self.current_position[1]
        
        # Get odor reading from predictor
        if self.predictor:
            self.current_odor = self.predictor.step_update(x, y, dt=self.dt)
        else:
            # Fallback if no predictor is available
            self.current_odor = 0.0
        
        # Check if we're in odor
        currently_in_odor = (self.current_odor >= self.odor_threshold)
        
        # Check for odor detection
        if currently_in_odor and not self.hit_occurred:
            rospy.loginfo(f"Odor detected at concentration {self.current_odor:.2f}")
            self.add_whiff_marker(x, y, self.current_position[2])
            self.hit_occurred = True
            self.last_hit_time = current_time
            self.is_surging = True
            self.is_casting = False
            self.surge_end_time = current_time + self.surge_duration
        elif not currently_in_odor:
            self.hit_occurred = False
        
        # Check if surge time is over
        if current_time > self.surge_end_time and self.is_surging and not currently_in_odor:
            rospy.loginfo("Surge time over, switching to casting")
            self.is_surging = False
            self.is_casting = True
        
        # Compute velocity commands
        vx, vy, vz = 0.0, 0.0, 0.0
        
        if self.is_surging:
            # Surge behavior: move upwind (negative X direction)
            vx = -self.surge_speed
            vy = 0.0
            rospy.loginfo_throttle(1.0, "SURGING")
        else:
            # Cast behavior: sinusoidal pattern
            time_since_hit = current_time - self.last_hit_time
            
            # Compute crosswind amplitude (grows with time)
            cast_amp = min(self.max_cast_amplitude, 
                          self.cast_growth_rate * time_since_hit)
            
            # Sinusoidal crosswind movement
            vy = cast_amp * math.sin(2.0 * math.pi * self.cast_base_freq * time_since_hit)
            
            # Keep moving upwind but at lower speed
            vx = -self.base_speed
            
            rospy.loginfo_throttle(1.0, f"CASTING with amplitude {cast_amp:.2f}")
        
        # Altitude control to maintain target height
        error_z = self.altitude - self.current_position[2]
        vz = 0.5 * error_z  # P controller for altitude
        
        # Only check if we're close to the source if we've confirmed reaching the starting position
        if self.reached_start_position:
            dist_to_source = np.linalg.norm(self.target_pos - self.current_position[:2])
            if dist_to_source < self.closest_to_source:
                rospy.loginfo(f"Source reached at {self.current_position[:2]}")
                vx, vy, vz = 0.0, 0.0, 0.0
                # Hover at the source
                self.publish_velocity(0.0, 0.0, 0.0)
                rospy.signal_shutdown("Source reached successfully")
                return
        
        # Send velocity commands
        self.publish_velocity(vx, vy, vz)
    
    def publish_velocity(self, vx, vy, vz):
        """Publish velocity commands"""
        # Create TwistStamped message for velocity
        from geometry_msgs.msg import TwistStamped
        vel_cmd = TwistStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        vel_cmd.header.frame_id = "base_link"
        vel_cmd.twist.linear.x = vx
        vel_cmd.twist.linear.y = vy
        vel_cmd.twist.linear.z = vz
        self.velocity_pub.publish(vel_cmd)
        
        # Also publish position target for better PX4 control
        pt = PositionTarget()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = (
            PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW
        )
        pt.velocity.x = vx
        pt.velocity.y = vy
        pt.velocity.z = vz
        
        # Set yaw to face direction of travel
        speed_xy = math.sqrt(vx*vx + vy*vy)
        if speed_xy > 0.1:
            # Calculate desired yaw
            desired_yaw = math.atan2(vy, vx)
            pt.yaw_rate = 0.5 * self.angle_difference(desired_yaw, self.current_yaw)
        else:
            pt.yaw_rate = 0.0
        
        self.setpoint_pub.publish(pt)
    
    @staticmethod
    def angle_difference(a, b):
        """Calculate the shortest angle difference between two angles"""
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def takeoff_sequence(self):
        """Execute initial takeoff sequence"""
        rospy.loginfo("Beginning takeoff sequence")
        
        # First, just reach minimal altitude at current position
        initial_takeoff = PositionTarget()
        initial_takeoff.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        initial_takeoff.type_mask = (
            PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW_RATE
        )
        # Take off at current XY position first - 
        # Important: use actual current position instead of 0,0 to avoid
        # false detection of being at the target immediately
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        
        # Use a slight offset from zero if we're too close to the origin
        if abs(current_x) < 0.1 and abs(current_y) < 0.1:
            current_x = 1.0  # Small offset to avoid being at "source"
            current_y = 1.0
        
        initial_takeoff.position.x = current_x
        initial_takeoff.position.y = current_y
        initial_takeoff.position.z = 2  # Minimal altitude to clear ground
        initial_takeoff.yaw = 0
        
        # Only after reaching altitude, move to starting position
        # Set target position for tracking
        takeoff_pose = PositionTarget()
        takeoff_pose.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        takeoff_pose.type_mask = (
            PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW_RATE
        )
        takeoff_pose.position.x = self.start_x
        takeoff_pose.position.y = self.start_y
        takeoff_pose.position.z = self.altitude
        takeoff_pose.yaw = math.pi  # Face in negative X direction (toward source)
        
        # Send a few setpoints before starting
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.setpoint_pub.publish(initial_takeoff)
            self.rate.sleep()
        
        # Set mode to OFFBOARD
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        
        # Arm the vehicle
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        
        last_req = rospy.Time.now()
        
        # Keep trying until OFFBOARD and armed
        while not rospy.is_shutdown() and (not self.current_state.armed or self.current_state.mode != "OFFBOARD"):
            self.setpoint_pub.publish(initial_takeoff)
            
            # Try to set OFFBOARD mode
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
                response = self.set_mode_client.call(offb_set_mode)
                if response.mode_sent:
                    rospy.loginfo("OFFBOARD enabled")
                else:
                    rospy.logwarn("Failed to set OFFBOARD mode")
                last_req = rospy.Time.now()
            
            # Try to arm
            if not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
                response = self.arming_client.call(arm_cmd)
                if response.success:
                    rospy.loginfo("Vehicle armed")
                else:
                    rospy.logwarn("Failed to arm vehicle")
                last_req = rospy.Time.now()
            
            self.rate.sleep()
        
        # Wait for basic takeoff altitude to be reached
        rospy.loginfo("Waiting to reach initial takeoff altitude...")
        takeoff_start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            current_alt = self.current_position[2]
            error = abs(current_alt - 2.0)  # Initial altitude is 2m
            
            if error < 0.3:  # Within 30cm
                rospy.loginfo(f"Reached initial altitude of 2.0m")
                break
                
            # Timeout after 20 seconds
            if (rospy.Time.now() - takeoff_start_time) > rospy.Duration(20.0):
                rospy.logwarn("Timeout waiting for takeoff. Continuing anyway.")
                break
            
            self.setpoint_pub.publish(initial_takeoff)
            self.rate.sleep()
            
        # Now move to the actual starting position
        rospy.loginfo(f"Moving to starting position ({self.start_x}, {self.start_y}, {self.altitude})")
        
        # Send the target position for a while to ensure the controller picks it up
        for _ in range(100):  # Reduced from 200 to 100
            if rospy.is_shutdown():
                break
            self.setpoint_pub.publish(takeoff_pose)
            self.rate.sleep()
        
        rospy.loginfo("Starting position commands sent. Will begin tracking when we reach the vicinity.")
        
        # Don't wait here - the tracking_step function will handle waiting to reach the position
        # This prevents any premature "source reached" detections
        # The tracking_step will continue sending position commands until we reach the starting area
        
        rospy.loginfo("Takeoff complete, beginning odor tracking")
    
    def run(self):
        """Main loop"""
        # Wait for connection to the flight controller
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.loginfo_throttle(1.0, "Waiting for FCU connection...")
            self.rate.sleep()
        
        rospy.loginfo("Connected to FCU")
        
        # Execute takeoff sequence
        self.takeoff_sequence()
        
        # Main control loop
        while not rospy.is_shutdown():
            # Main work is done in the callback
            self.rate.sleep()

if __name__ == "__main__":
    try:
        controller = OdorTrackerNode()
        controller.run()
    except rospy.ROSInterruptException:
        pass
#! /usr/bin/env python

import rospy
import csv
import sys
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from visualization_msgs.msg import Marker, MarkerArray
import math
import time
import tf.transformations
import os
import datetime  # For timestamped log filename
from std_msgs.msg import Float32  # <-- Added import for odor publication

# For odor detection - assuming a similar implementation to the example
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
        self.prev_heading = 0.0 
        
        # Flag to track if we've reached the starting position
        self.reached_start_position = False
        
        # Starting position (well inside the plume)
        self.start_x = 35.0
        self.start_y = 6.0
        
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
        self.surge_speed = 2.5  # m/s during surge
        self.base_speed = 0.5   # m/s during normal flight
        self.surge_duration = 2.0  # seconds to continue surge after losing odor
        self.surge_end_time = 0
        
        # Casting parameters
        self.cast_base_freq = 0.5   # Base frequency for sine wave (Hz)
        self.cast_growth_rate = 0.5  # Rate at which amplitude grows per second out of odor
        self.max_cast_amplitude = 12.0  # Maximum crosswind amplitude
        
        # Altitude to maintain
        self.altitude = 4.0  # meters
        self.tracking_start_position = None
        self.last_hit_position = None
        
        # ------------- Logging Setup -------------
        self.data_log = []
        self.log_columns = [
            "timestamp", "x", "y", "z", "yaw", "odor_concentration",
            "is_surging", "is_casting", "target_x", "target_y", "target_z", 
            "heading", "crosswind_offset", "cast_amplitude", 
            "vx", "vy", "vz", "angular_velocity"
        ]
        self.log_dir = os.path.expanduser("/home/vbl/gazebo_ws/src/plume_tracking_logs/odor_tracking_logs/")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # Use a human-readable timestamp for a unique filename (e.g., YYYYMMDD_HHMMSS)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_dir, f"odor_tracking_pos_{timestamp_str}.csv")
        # Write the CSV header
        with open(self.log_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.log_columns)
        rospy.loginfo(f"Logging data to: {self.log_filename}")
        rospy.on_shutdown(self.save_log)
        # -----------------------------------------
        
        # Load odor predictor if available
        self.load_odor_predictor()
        
        # ROS subscribers
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_callback)
        self.pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.pose_callback)
        
        # ROS publishers
        self.setpoint_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        from geometry_msgs.msg import TwistStamped
        self.velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
        self.drone_marker_pub = rospy.Publisher("drone_position_marker", Marker, queue_size=10)
        
        # Publish the odor concentration
        self.odor_pub = rospy.Publisher("odor_concentration", Float32, queue_size=10)
        
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
                fitted_p_heatmap=hmap_data['fitted_heatmap']*3,  # Multiplied by 5 as in velocity controller
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
        self.markers = MarkerArray()
        
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
                # Store the position where we start tracking
                self.tracking_start_position = self.current_position.copy()
                self.last_hit_position = self.current_position.copy()
                self.last_hit_time = rospy.Time.now().to_sec()
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
        
        # ---------------- Publish odor reading here ----------------
        self.odor_pub.publish(Float32(self.current_odor))
        
        # Check if we're in odor
        currently_in_odor = (self.current_odor >= self.odor_threshold)
        
        # Check for odor detection
        if currently_in_odor and not self.hit_occurred:
            rospy.loginfo(f"Odor detected at concentration {self.current_odor:.2f}")
            self.add_whiff_marker(x, y, self.current_position[2])
            self.hit_occurred = True
            self.last_hit_time = current_time
            self.last_hit_position = self.current_position.copy()  # Store position where odor was detected
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
            
            # Calculate theoretical position based on surge movement
            elapsed_time = current_time - self.last_hit_time
            upwind_distance = self.surge_speed * elapsed_time
            
            # Create a new reference position that maintains forward progress
            new_reference_x = self.last_hit_position[0] - upwind_distance  # negative X direction
            new_reference_y = self.last_hit_position[1]  # Maintain Y position
            
            # Store new reference point and reset timer
            self.last_hit_position = np.array([new_reference_x, new_reference_y, self.altitude])
            self.last_hit_time = current_time
                
        # Position target
        pt = PositionTarget()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = (
            PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW_RATE
        )
        
        # Variables for logging
        target_x, target_y, target_z = 0.0, 0.0, 0.0
        heading = 0.0
        crosswind_offset = 0.0
        cast_amplitude = 0.0
        vx, vy, vz = 0.0, 0.0, 0.0
        
        if self.is_surging:
            # Surge behavior: move upwind (negative X direction) using position control
            elapsed_time = current_time - self.last_hit_time
            upwind_distance = self.surge_speed * elapsed_time
            
            target_x = self.last_hit_position[0] - upwind_distance
            target_y = self.last_hit_position[1]
            target_z = self.altitude
            
            pt.position.x = target_x
            pt.position.y = target_y
            pt.position.z = target_z
            pt.yaw = math.pi  # Face upwind
            
            heading = math.pi
            
            vx = -self.surge_speed
            vy = 0.0
            vz = 0.0
            
            rospy.loginfo_throttle(1.0, "SURGING")
        else:
            # Cast behavior: sinusoidal pattern using position control
            time_since_cast_start = current_time - self.last_hit_time
            
            # Enhanced casting amplitude growth
            if time_since_cast_start < 3.0:  # first 3 seconds
                cast_amplitude = self.cast_growth_rate * 2 * time_since_cast_start
            else:
                cast_amplitude = min(self.max_cast_amplitude, 
                                     self.cast_growth_rate * time_since_cast_start)
            
            freq_factor = 1.0 / (1.0 + 0.009 * time_since_cast_start)
            current_cast_freq = self.cast_base_freq * freq_factor
            
            cast_phase = math.sin(2.0 * math.pi * current_cast_freq * time_since_cast_start)
            crosswind_offset = cast_amplitude * cast_phase
            
            upwind_distance = self.base_speed * time_since_cast_start
            
            target_x = self.last_hit_position[0] - upwind_distance
            target_y = self.last_hit_position[1] + crosswind_offset
            target_z = self.altitude
            
            pt.position.x = target_x
            pt.position.y = target_y
            pt.position.z = target_z
            
            heading = math.atan2(crosswind_offset, -upwind_distance)
            pt.yaw = heading
            
            vx = -self.base_speed
            
            crosswind_velocity = (cast_amplitude * 2.0 * math.pi * current_cast_freq * 
                                  math.cos(2.0 * math.pi * current_cast_freq * time_since_cast_start))
            
            if time_since_cast_start < 3.0:
                amplitude_growth_rate = self.cast_growth_rate * 2
            else:
                if cast_amplitude < self.max_cast_amplitude:
                    amplitude_growth_rate = self.cast_growth_rate
                else:
                    amplitude_growth_rate = 0.0
            
            amplitude_velocity_component = amplitude_growth_rate * cast_phase
            vy = crosswind_velocity + amplitude_velocity_component
            
            error_z = self.altitude - self.current_position[2]
            vz = 0.5 * error_z
            
            rospy.loginfo_throttle(0.5, f"time={time_since_cast_start:.2f}s, amp={cast_amplitude:.2f}, "
                                     f"offset={crosswind_offset:.2f}, sin_val={cast_phase:.2f}")
            rospy.loginfo_throttle(1.0, f"CASTING with amplitude {cast_amplitude:.2f}")
        
        # Calculate angular velocity as difference between current heading and previous heading
        angle_diff = math.atan2(math.sin(heading - self.prev_heading),
                                math.cos(heading - self.prev_heading))
        angular_velocity = angle_diff / self.dt
        self.prev_heading = heading
        
        # Log the data
        self.log_data(target_x, target_y, target_z, heading, crosswind_offset, 
                      cast_amplitude, vx, vy, vz, angular_velocity)
        
        # Also publish velocity for mixed control mode
        self.publish_velocity(vx, vy, vz, heading)
        
        # Check if we're close to the source (only after start reached)
        if self.reached_start_position:
            dist_to_source = np.linalg.norm(self.target_pos - self.current_position[:2])
            if dist_to_source < self.closest_to_source:
                rospy.loginfo(f"Source reached at {self.current_position[:2]}")
                
                # Hover briefly
                hover_pt = PositionTarget()
                hover_pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                hover_pt.type_mask = (
                    PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
                    PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
                    PositionTarget.IGNORE_YAW_RATE
                )
                hover_pt.position.x = self.current_position[0]
                hover_pt.position.y = self.current_position[1]
                hover_pt.position.z = self.altitude
                
                hover_start_time = rospy.Time.now()
                while (rospy.Time.now() - hover_start_time) < rospy.Duration(3.0):
                    self.setpoint_pub.publish(hover_pt)
                    self.rate.sleep()
                
                # Save any remaining log data
                self.save_log()
                
                # Reset and continue
                self.last_hit_time = rospy.Time.now().to_sec()
                self.last_hit_position = self.current_position.copy()
                self.is_casting = True
                self.is_surging = False
                
                return
        
        # Send position command
        self.setpoint_pub.publish(pt)
    
    def log_data(self, target_x, target_y, target_z, heading, crosswind_offset, 
                 cast_amplitude, vx, vy, vz, angular_velocity):
        """Append current state data to the log and write to file periodically."""
        timestamp = rospy.Time.now().to_sec()
        log_entry = [
            timestamp,
            self.current_position[0],
            self.current_position[1],
            self.current_position[2],
            self.current_yaw,
            self.current_odor,
            self.is_surging,
            self.is_casting,
            target_x, 
            target_y, 
            target_z,
            heading,
            crosswind_offset,
            cast_amplitude,
            vx,
            vy,
            vz,
            angular_velocity
        ]
        self.data_log.append(log_entry)
        # Write to file every 50 entries
        if len(self.data_log) >= 50:
            with open(self.log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_log)
            self.data_log = []
    
    def save_log(self):
        """Write any remaining log entries to file on shutdown."""
        if self.data_log:
            rospy.loginfo(f"Saving {len(self.data_log)} log entries to {self.log_filename}")
            with open(self.log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_log)
    
    def publish_velocity(self, vx, vy, vz, heading):
        """Publish velocity commands alongside position commands for better control"""
        from geometry_msgs.msg import TwistStamped
        vel_cmd = TwistStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        vel_cmd.header.frame_id = "base_link"
        vel_cmd.twist.linear.x = vx
        vel_cmd.twist.linear.y = vy
        vel_cmd.twist.linear.z = vz
        self.velocity_pub.publish(vel_cmd)
        
        vel_target = PositionTarget()
        vel_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        vel_target.type_mask = (
            PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW
        )
        vel_target.velocity.x = vx
        vel_target.velocity.y = vy
        vel_target.velocity.z = vz
        
        speed_xy = math.sqrt(vx*vx + vy*vy)
        if speed_xy > 0.1:
            desired_yaw = math.atan2(vy, vx)
            vel_target.yaw_rate = 0.5 * self.angle_difference(desired_yaw, self.current_yaw)
        else:
            vel_target.yaw_rate = 0.0
    
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
        
        initial_takeoff = PositionTarget()
        initial_takeoff.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        initial_takeoff.type_mask = (
            PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ +
            PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ +
            PositionTarget.IGNORE_YAW_RATE
        )
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        
        if abs(current_x) < 0.1 and abs(current_y) < 0.1:
            current_x = 1.0
            current_y = 1.0
        
        initial_takeoff.position.x = current_x
        initial_takeoff.position.y = current_y
        initial_takeoff.position.z = 2
        initial_takeoff.yaw = 0
        
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
        takeoff_pose.yaw = math.pi  # Face negative X direction
        
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.setpoint_pub.publish(initial_takeoff)
            self.rate.sleep()
        
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        
        last_req = rospy.Time.now()
        
        while not rospy.is_shutdown() and (not self.current_state.armed or self.current_state.mode != "OFFBOARD"):
            self.setpoint_pub.publish(initial_takeoff)
            
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
                response = self.set_mode_client.call(offb_set_mode)
                if response.mode_sent:
                    rospy.loginfo("OFFBOARD enabled")
                else:
                    rospy.logwarn("Failed to set OFFBOARD mode")
                last_req = rospy.Time.now()
            
            if not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
                response = self.arming_client.call(arm_cmd)
                if response.success:
                    rospy.loginfo("Vehicle armed")
                else:
                    rospy.logwarn("Failed to arm vehicle")
                last_req = rospy.Time.now()
            
            self.rate.sleep()
        
        rospy.loginfo("Waiting to reach initial takeoff altitude...")
        takeoff_start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            current_alt = self.current_position[2]
            error = abs(current_alt - 2.0)
            
            if error < 0.3:
                rospy.loginfo(f"Reached initial altitude of 2.0m")
                break
                
            if (rospy.Time.now() - takeoff_start_time) > rospy.Duration(20.0):
                rospy.logwarn("Timeout waiting for takeoff. Continuing anyway.")
                break
            
            self.setpoint_pub.publish(initial_takeoff)
            self.rate.sleep()
            
        rospy.loginfo(f"Moving to starting position ({self.start_x}, {self.start_y}, {self.altitude})")
        
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.setpoint_pub.publish(takeoff_pose)
            self.rate.sleep()
        
        rospy.loginfo("Starting position commands sent. Will begin tracking when we reach the vicinity.")
        rospy.loginfo("Takeoff complete, beginning odor tracking")
    
    def run(self):
        """Main loop"""
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.loginfo_throttle(1.0, "Waiting for FCU connection...")
            self.rate.sleep()
        
        rospy.loginfo("Connected to FCU")
        
        self.takeoff_sequence()
        
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == "__main__":
    try:
        args = rospy.myargv(argv=sys.argv)
        if len(args) >= 3:
            start_x = float(args[1])
            start_y = float(args[2])
        else:
            rospy.logwarn("Start location not provided. Using default values.")
            start_x = 40.0
            start_y = 10.0

        controller = OdorTrackerNode()
        # Override the default start position with terminal inputs:
        controller.start_x = start_x
        controller.start_y = start_y
        controller.run()
    except rospy.ROSInterruptException:
        pass

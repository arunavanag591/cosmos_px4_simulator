#!/usr/bin/env python
import rospy
import numpy as np
import math
import csv
import os
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.msg import State, PositionTarget
from gazebo_px4_simulator.msg import OdorReading
import tf.transformations

# Import the required modules directly
from plume_tracking import SurgeCastAgent
from plume_sim_fast import CosmosFast
import pandas as pd

class UnifiedPlumeTrackerNode:
    def __init__(self):
        rospy.init_node('unified_plume_tracker')
        
        # Load parameters
        # self.bounds = [(0, 50), (-15, 15)]  # Default bounds
        # self.start_pos = np.array([25.0, 6.0])  # Starting position
        # self.target_pos = np.array([0.0, 0.0])  # Target (odor source) position
        # self.target_weight = 0.1  # Weight for target direction vs. plume following
        # self.plume_timeout = 7.0  # Seconds before increasing target weight
        # self.closest_to_source = 0.5  # Distance threshold to consider target reached
        
        ## rigolli bounds
        self.bounds = [(5, 40), (-0, 8)]  # Default bounds
        self.start_pos = np.array([20.0, 6.0])  # Starting position
        self.target_pos = np.array([5.0, 4.0])  # Target (odor source) position
        self.target_weight = 0.3  # Weight for target direction vs. plume following
        self.plume_timeout = 10.0  # Seconds before increasing target weight
        self.closest_to_source = 0.5  # Distance threshold to consider target reached



        # Add backward boundary - don't go further than 1m behind the source
        self.backward_boundary = self.target_pos[0] - 1.0
        
        # Add return behavior states
        self.return_to_start = False  # Flag to trigger return behavior
        self.source_reached = False   # Flag to indicate source was reached
        self.crossed_source = False   # Flag to indicate drone crossed behind source
        
        # Set up data logging
        self.data_log = []
        self.log_columns = ["timestamp", "x", "y", "z", "yaw", "odor_concentration", 
                           "is_surging", "surge_force", "vx", "vy", "whiff_count",
                           "state", "return_to_start"]
        
        # Create output directory if it doesn't exist
        self.log_dir = os.path.expanduser("~/gazebo_ws/src/plume_tracking_logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create unique log filename with timestamp
        timestamp = rospy.Time.now().to_sec()
        self.log_filename = os.path.join(self.log_dir, f"plume_tracking_{int(timestamp)}.csv")
        
        # Open log file and write header
        with open(self.log_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.log_columns)
        
        rospy.loginfo(f"Logging data to: {self.log_filename}")
        
        # Load odor model data
        dirname = rospy.get_param('~odor_model_path', '/home/vbl/gazebo_ws/src/gazebo_px4_simulator/odor_sim_assets/rigolli/')
        rospy.loginfo(f"Loading odor model data from {dirname}")
        
        try:
            hmap_data = np.load(str(dirname) + "hmap.npz")
            fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
            fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')
            
            # Initialize the odor predictor
            self.predictor = CosmosFast(
                # fitted_p_heatmap=hmap_data['fitted_heatmap'],   #hws
                fitted_p_heatmap=hmap_data['fitted_p_heatmap'],   #rigolli
                xedges=hmap_data['xedges'],
                yedges=hmap_data['yedges'],
                fdf=fdf,
                fdf_nowhiff=fdf_nowhiff
            )
            rospy.loginfo("CosmosFast predictor initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Error loading odor model data: {e}")
            rospy.signal_shutdown("Failed to load odor model data")
            return
        
        # Initialize the surge cast agent with better casting parameters
        self.surge_agent = SurgeCastAgent(
            tau=0.3,            # Slightly slower dynamics for smoother movement
            noise=1.5,           # Keep noise level for random movements
            bias=0.5,            # Bias weight for movement
            threshold=4.5,       # Odor threshold for hit detection
            hit_trigger='peak',  # Type of hit detection
            surge_amp=3.5,       # Slightly reduced surge amplitude
            tau_surge=1,       # Shorter surge time constant for more casting
            cast_freq=1.0,       # Higher casting frequency
            cast_width=1.5,      # Increased casting width
            bounds=self.bounds
        )
        
        # State variables for tracking
        self.current_position = np.array([0.0, 0.0, 0.0])  # Current 3D position
        self.current_yaw = 0.0  # Current yaw angle
        self.current_odor = 0.0  # Current odor concentration
        self.v = np.zeros(2)    # Current velocity vector
        
        # Additional tracking variables for logging
        self.surge_active = False
        self.surge_force = 0.0
        self.whiff_count = 0
        
        self.dt = 0.01  # Time step for simulation (100Hz)
        self.is_initialized = False  # Flag to ensure we have valid position data
        
        # Current drone behavior state
        self.current_state = "SEARCHING"  # SEARCHING, SOURCE_REACHED, RETURNING
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', Twist, queue_size=10)
        self.setpoint_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.odor_pub = rospy.Publisher('odor_reading', OdorReading, queue_size=10)
        
        # Subscribers
        self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_callback)
        
        self.drone_state = None
        
        self.rate = rospy.Rate(100)  # 100Hz control loop
        rospy.loginfo("Unified plume tracker initialized")
        
        # Set up shutdown handler to ensure log is saved
        rospy.on_shutdown(self.save_log)
    
    def save_log(self):
        """Save any remaining log data when node shuts down"""
        if self.data_log:
            rospy.loginfo(f"Saving {len(self.data_log)} log entries to {self.log_filename}")
            with open(self.log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_log)
            self.data_log = []
    
    def log_data(self):
        """Add current state to data log"""
        # Create log entry
        entry = [
            rospy.Time.now().to_sec(),     # timestamp
            self.current_position[0],      # x
            self.current_position[1],      # y
            self.current_position[2],      # z
            self.current_yaw,              # yaw
            self.current_odor,             # odor_concentration
            self.surge_active,             # is_surging
            self.surge_force,              # surge_force
            self.v[0],                     # vx
            self.v[1],                     # vy
            self.whiff_count,              # whiff_count
            self.current_state,            # current behavior state
            self.return_to_start           # return flag
        ]
        
        # Add to in-memory log
        self.data_log.append(entry)
        
        # Write to file periodically to avoid data loss
        if len(self.data_log) >= 100:
            with open(self.log_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_log)
            self.data_log = []
    
    def state_callback(self, msg):
        """Store the current drone state"""
        self.drone_state = msg
    
    def pose_callback(self, msg):
        """Process drone position updates and run tracking algorithm"""
        # Extract 3D position and orientation from message
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
        
        # Set initialized flag once we have a valid position
        if not self.is_initialized and not np.all(self.current_position == 0):
            self.is_initialized = True
            rospy.loginfo(f"Tracker initialized at position: {self.current_position[:2]}")
        
        # Only proceed if initialized and drone is in OFFBOARD mode
        if self.is_initialized and self.drone_state and self.drone_state.mode == "OFFBOARD" and self.drone_state.armed:
            # Run a single step of the tracking algorithm
            self.tracking_step()
    
    def tracking_step(self):
        """Run a single step of the tracking algorithm"""
        # Get current position and time
        x = self.current_position[:2]  # 2D position (x, y)
        current_time = rospy.Time.now().to_sec()
        
        # Check if drone has crossed behind the source (x < backward_boundary)
        if x[0] < self.backward_boundary and not self.crossed_source:
            self.crossed_source = True
            self.return_to_start = True
            self.current_state = "RETURNING"
            rospy.loginfo(f"Drone crossed behind source at x={x[0]:.2f}, initiating return to start")
        
        # Calculate distance to target
        to_target = self.target_pos - x
        dist_to_target = np.linalg.norm(to_target)
        
        # Check if we've reached the target
        if dist_to_target < self.closest_to_source and not self.source_reached:
            self.source_reached = True
            self.return_to_start = True
            self.current_state = "SOURCE_REACHED"
            rospy.loginfo(f"Target reached at {x}, initiating return to start")
        
        # Return behavior logic
        if self.return_to_start:
            # Calculate direction to starting position
            to_start = self.start_pos - x
            dist_to_start = np.linalg.norm(to_start)
            
            # Check if we've reached the starting position
            if dist_to_start < 1.0:
                rospy.loginfo("Returned to starting position")
                self.return_to_start = False
                self.crossed_source = False
                self.source_reached = False
                self.current_state = "SEARCHING"
                # Reset surge and casting states
                self.surge_active = False
                self.surge_force = 0.0
                # Reset the surge agent for a fresh start
                self.surge_agent.reset()
            else:
                # Normalize direction to start
                to_start = to_start / dist_to_start
                
                # Set velocity toward start position with some randomness
                return_speed = 2.0  # Fixed return speed
                self.v = to_start * return_speed + np.random.normal(0, 0.1, 2)
                
                # Publish velocity
                self.publish_velocity()
                
                # Log data during return
                self.log_data()
                return
        
        # Get odor at current position
        self.current_odor = self.predictor.step_update(x[0], x[1], dt=self.dt)
        
        # Publish odor reading for visualization
        odor_msg = OdorReading()
        odor_msg.header.stamp = rospy.Time.now()
        odor_msg.concentration = self.current_odor
        odor_msg.position.x = x[0]
        odor_msg.position.y = x[1]
        self.odor_pub.publish(odor_msg)
        
        # Use the SurgeCastAgent step method to get the next action
        step_result = self.surge_agent.step(
            x, self.current_odor, current_time, self.target_pos, 
            self.target_weight, self.plume_timeout, self.dt
        )
        
        # Update our state variables from the step result
        self.v = step_result['v']
        self.surge_active = step_result['surge_active']
        self.surge_force = step_result['surge_force']
        self.whiff_count = step_result['whiff_count']
        
        # Set current state based on agent status
        if self.surge_active:
            self.current_state = "SURGING"
        else:
            self.current_state = "CASTING"
        
        # Additional boundary check - don't go too far behind source
        if x[0] < self.backward_boundary:
            # Force positive x velocity to move away from boundary
            self.v[0] = max(self.v[0], 0.5)
            
        # Log current data
        self.log_data()
        
        # Log status periodically
        if int(current_time * 10) % 10 == 0:  # Log approximately once per second
            behavior = "SURGING" if self.surge_active else "CASTING"
            rospy.loginfo(f"Behavior: {behavior}, Whiffs: {self.whiff_count}")
            rospy.loginfo(f"Position: ({x[0]:.2f}, {x[1]:.2f}), Velocity: ({self.v[0]:.2f}, {self.v[1]:.2f})")
            rospy.loginfo(f"Odor concentration: {self.current_odor:.2f}")
            rospy.loginfo(f"Surge active: {self.surge_active}, Surge force: {self.surge_force:.2f}")
        
        # Publish velocity command
        self.publish_velocity()
    
    def publish_velocity(self, velocity=None):
        """Publish velocity command to move the drone"""
        if velocity is None:
            velocity = self.v
            
        # Create Twist message with current velocity
        vel_cmd = Twist()
        vel_cmd.linear.x = velocity[0]
        vel_cmd.linear.y = velocity[1]
        vel_cmd.linear.z = 0.0  # Maintain altitude
        
        # Publish velocity command
        self.cmd_vel_pub.publish(vel_cmd)
        
        # Also publish position target for better PX4 control
        position_target = PositionTarget()
        position_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        position_target.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ + \
                                  PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ + \
                                  PositionTarget.IGNORE_YAW_RATE
        
        # Set look-ahead position based on velocity
        position_target.position.x = self.current_position[0] + velocity[0] * 0.5  # Look ahead 0.5 seconds
        position_target.position.y = self.current_position[1] + velocity[1] * 0.5
        position_target.position.z = 3.0  # Fixed altitude
        
        # Calculate yaw to face direction of travel
        if np.linalg.norm(velocity) > 0.1:
            position_target.yaw = math.atan2(velocity[1], velocity[0])
        else:
            position_target.yaw = 0.0
        
        # Publish position target
        self.setpoint_pub.publish(position_target)
    
    def run(self):
        """Main loop for the node"""
        rospy.loginfo("Starting unified plume tracker")
        while not rospy.is_shutdown():
            # If we're initialized but not moving, start with a small velocity
            if self.is_initialized and np.linalg.norm(self.v) < 0.1 and not self.return_to_start:
                now = rospy.Time.now().to_sec()
                cast_phase = np.sin(2*np.pi*0.5*now)
                # Small initial velocity to start moving
                self.v = np.array([-0.2, cast_phase * 0.4])
                self.publish_velocity()
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        tracker = UnifiedPlumeTrackerNode()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
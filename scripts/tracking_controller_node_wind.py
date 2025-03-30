#!/usr/bin/env python
import rospy
import numpy as np
import math
import csv
import os
from geometry_msgs.msg import Twist, PoseStamped, Vector3
from mavros_msgs.msg import State, PositionTarget
from gazebo_px4_simulator.msg import OdorReading
from std_msgs.msg import Float32
import tf.transformations

# Import the required modules directly
from plume_tracking import SurgeCastAgent
from plume_sim_fast import CosmosFast
import pandas as pd

class UnifiedPlumeTrackerNode:
    def __init__(self):
        rospy.init_node('unified_plume_tracker')
        
        # # Load parameters
        # self.bounds = [(0, 50), (-25, 25)]  # Default bounds
        # self.start_pos = np.array([15.0, 6.0])  # Starting position
        # self.target_pos = np.array([0.0, 0.0])  # Target (odor source) position
        # self.target_weight = 0.1  # Weight for target direction vs. plume following
        # self.plume_timeout = 10.0  # Seconds before increasing target weight
        # self.closest_to_source = 0.5  # Distance threshold to consider target reached
        
        ## rigolli bounds
        self.bounds = [(5, 40), (-0, 8)]  # Default bounds
        self.start_pos = np.array([25.0, 6.0])  # Starting position
        self.target_pos = np.array([5.0, 4.0])  # Target (odor source) position
        self.target_weight = 0.1  # Weight for target direction vs. plume following
        self.plume_timeout = 10.0  # Seconds before increasing target weight
        self.closest_to_source = 0.5  # Distance threshold to consider target reached

        # Wind estimation variables
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.wind_z = 0.0
        self.wind_magnitude = 0.0

        # Set up data logging
        self.data_log = []
        self.log_columns = ["timestamp", "x", "y", "z", "yaw", "odor_concentration", 
                           "is_surging", "surge_force", "vx", "vy", "whiff_count",
                           "wind_x", "wind_y", "wind_z", "wind_magnitude"]  # Added wind columns
        
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
        # dirname = rospy.get_param('~odor_model_path', '/home/vbl/gazebo_ws/src/gazebo_px4_simulator/odor_sim_assets/hws/')
        dirname = rospy.get_param('~odor_model_path', '/home/vbl/gazebo_ws/src/gazebo_px4_simulator/odor_sim_assets/rigolli/')
        rospy.loginfo(f"Loading odor model data from {dirname}")
        
        try:
            hmap_data = np.load(str(dirname) + "hmap.npz")
            fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
            fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')
            
            # Initialize the odor predictor
            self.predictor = CosmosFast(
                fitted_p_heatmap=hmap_data['fitted_p_heatmap'],   #hws
                # fitted_p_heatmap=hmap_data['fitted_p_heatmap'], #rigolli
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
        
        # Initialize the surge cast agent with stronger surge parameters
        self.surge_agent = SurgeCastAgent(
            tau=0.3,            
            noise=3,           
            bias=0.1,            # Increase bias for stronger movement
            threshold=6.5,       # Slightly lower threshold for more hits
            hit_trigger='peak',  
            surge_amp=4.0,       # Stronger surge
            tau_surge=1,       # Shorter surge for more frequent casting
            cast_freq=1,       # Higher frequency casting
            cast_width=0.8,      # Much wider casting pattern
            bounds=self.bounds
        )
        
        # State variables for tracking
        self.current_position = np.array([0.0, 0.0, 0.0])  # Current 3D position
        self.current_yaw = 0.0  # Current yaw angle
        self.current_odor = 0.0  # Current odor concentration
        self.last_odor = 0.0  # Previous odor concentration for peak detection
        self.hit_occurred = False  # Flag for odor peak detection
        self.last_hit_time = -np.inf  # Time of last odor hit
        self.v = np.zeros(2)  # Current velocity vector
        self.b = np.zeros(2)  # Current bias force
        
        # Surge variables - using simple approach with direct calculation
        self.surge_active = False
        self.surge_start_time = 0
        self.surge_duration = 5.0  # Duration of surge behavior in seconds
        self.surge_force = 0.0     # Current surge force value
        
        self.dt = 0.005  # Time step for simulation (100Hz)
        self.is_initialized = False  # Flag to ensure we have valid position data
        self.whiff_count = 0  # Count of detected whiffs
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', Twist, queue_size=10)
        self.setpoint_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.odor_pub = rospy.Publisher('odor_reading', OdorReading, queue_size=10)
        
        # Subscribers
        self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_callback)
        
        # Subscribe to wind data
        self.wind_sub = rospy.Subscriber('/estimated_wind', Vector3, self.wind_cb)
        self.wind_mag_sub = rospy.Subscriber('/wind_magnitude', Float32, self.wind_mag_cb)
        
        self.drone_state = None
        
        self.rate = rospy.Rate(200)  # 200Hz control loop
        rospy.loginfo("Unified plume tracker initialized")
        
        # Set up shutdown handler to ensure log is saved
        rospy.on_shutdown(self.save_log)
    
    def wind_cb(self, msg):
        """Callback for wind vector data"""
        self.wind_x = msg.x
        self.wind_y = msg.y
        self.wind_z = msg.z

    def wind_mag_cb(self, msg):
        """Callback for wind magnitude data"""
        self.wind_magnitude = msg.data
    
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
        # Create log entry with wind data
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
            self.wind_x,                   # wind_x (new)
            self.wind_y,                   # wind_y (new)
            self.wind_z,                   # wind_z (new)
            self.wind_magnitude            # wind_magnitude (new)
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
        
        # Get odor at current position
        self.current_odor = self.predictor.step_update(x[0], x[1], dt=self.dt)
        
        # Publish odor reading for visualization
        odor_msg = OdorReading()
        odor_msg.header.stamp = rospy.Time.now()
        odor_msg.concentration = self.current_odor
        odor_msg.position.x = x[0]
        odor_msg.position.y = x[1]
        self.odor_pub.publish(odor_msg)
        
        # Detect odor hits (peaks)
        if self.surge_agent.hit_trigger == 'peak':
            if self.current_odor >= self.surge_agent.threshold:
                if self.current_odor <= self.last_odor and not self.hit_occurred:
                    # Hit detected!
                    self.whiff_count += 1
                    rospy.loginfo(f"Hit #{self.whiff_count} detected! Concentration: {self.current_odor:.2f}")
                    self.hit_occurred = True
                    
                    # Start a new surge
                    self.surge_active = True
                    self.surge_start_time = current_time
                    self.last_hit_time = current_time
                    self.surge_force = 10.0  # Start with strong surge force
                    
                    rospy.loginfo("SURGING initiated with force 10.0")
                
                self.last_odor = self.current_odor
            else:
                self.last_odor = 0
                self.hit_occurred = False
        
        # Calculate control inputs
        eta = np.random.normal(0, self.surge_agent.noise, 2)
        time_since_hit = current_time - self.last_hit_time
        
        # Check if surge is active
        if self.surge_active:
            surge_elapsed = current_time - self.surge_start_time
            if surge_elapsed < self.surge_duration:
                # Calculate surge force with exponential decay
                t = surge_elapsed
                self.surge_force = self.surge_agent.surge_amp_ * t * np.exp(-t/self.surge_agent.tau_surge)
                # Keep minimum surge force to maintain behavior
                self.surge_force = max(self.surge_force, 2.0)
            else:
                # End of surge
                self.surge_active = False
                self.surge_force = 0.0
        
        # Calculate direction to target
        to_target = self.target_pos - x
        dist_to_target = np.linalg.norm(to_target)
        
        # Check if we've reached the target

        if dist_to_target < self.closest_to_source:
            rospy.loginfo(f"Target reached at {x}")
            self.v = np.zeros(2)
            self.publish_velocity()
            
            # Log data at target
            self.log_data()

            # Add these lines to properly close the node
            rospy.loginfo("Target reached, shutting down node")
            # Ensure all logs are saved
            self.save_log()
            # Request node shutdown
            rospy.signal_shutdown("Target reached")

            return
        
        # Normalize target direction
        to_target = to_target / (dist_to_target + 1e-6)
        
        # Adjust target weight based on time since last hit
        current_target_weight = self.target_weight
        if time_since_hit > self.plume_timeout:
            current_target_weight = min(0.8, 
                self.target_weight + 0.1*(time_since_hit - self.plume_timeout)/self.plume_timeout)
        
        # Compute bias force based on current state
        if self.surge_active and self.surge_force > 1.0:
            # Surge behavior
            surge_direction = np.array([-1.0, -0.05*x[1]])
            surge_direction /= np.linalg.norm(surge_direction)
            self.b = (1 - current_target_weight)*surge_direction + current_target_weight*to_target
            self.b *= self.surge_force
            behavior = "SURGING"
        else:
            # Casting behavior
            cast_freq = 0.5
            cast_phase = np.sin(2*np.pi*cast_freq*current_time)
            base_cast_width = 1.0
            dist_factor = min(1.0, dist_to_target/20.0)
            cast_width = base_cast_width*dist_factor

            crosswind = np.array([0.0, cast_phase*cast_width])
            upwind = np.array([-0.2, 0.0])
            self.b = (1 - current_target_weight)*(upwind + crosswind) + current_target_weight*to_target
            norm_b = np.linalg.norm(self.b)
            if norm_b > 0:
                self.b *= self.surge_agent.bias/norm_b
            behavior = "CASTING"
        
        # Update velocity using AR dynamics
        self.v += (self.dt/self.surge_agent.tau)*(-self.v + eta + self.b)
        
        # Apply boundary conditions
        self.v, _ = self.surge_agent.reflect_if_out_of_bounds(self.v, x)
        
        # Scale velocity for stronger movement
        velocity_scale = 1.0
        if behavior == "SURGING":
            velocity_scale = 2.0  # Stronger movement during surging
        
        # Apply velocity scaling
        scaled_v = self.v * velocity_scale
        
        # Log current data
        self.log_data()
        
        # Log status periodically
        if int(current_time * 10) % 10 == 0:  # Log approximately once per second
            rospy.loginfo(f"Behavior: {behavior}, Whiffs: {self.whiff_count}")
            rospy.loginfo(f"Position: ({x[0]:.2f}, {x[1]:.2f}), Velocity: ({scaled_v[0]:.2f}, {scaled_v[1]:.2f})")
            rospy.loginfo(f"Odor concentration: {self.current_odor:.2f}")
            rospy.loginfo(f"Surge active: {self.surge_active}, Surge force: {self.surge_force:.2f}")
            # Add wind logging
            rospy.loginfo(f"Wind: ({self.wind_x:.2f}, {self.wind_y:.2f}, {self.wind_z:.2f}), Magnitude: {self.wind_magnitude:.2f}")
        
        # Publish scaled velocity command
        self.publish_velocity(scaled_v)
    
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
            if self.is_initialized and np.linalg.norm(self.v) < 0.1:
                now = rospy.Time.now().to_sec()
                cast_phase = np.sin(2*np.pi*0.5*now)
                # Small initial velocity to start moving
                self.v = np.array([-0.2, cast_phase * 0.4])  # Increased initial velocity
                self.publish_velocity()
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        tracker = UnifiedPlumeTrackerNode()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
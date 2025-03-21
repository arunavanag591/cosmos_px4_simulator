#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from gazebo_px4_simulator.msg import OdorReading
from gazebo_msgs.msg import ModelState
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from std_msgs.msg import ColorRGBA
import math

class GazeboVisualizer:
    def __init__(self):
        rospy.init_node('gazebo_plume_visualizer')
        
        # Parameters
        self.target_pos = rospy.get_param('~target_pos', [0.0, 0.0, 2.0])
        self.odor_threshold = rospy.get_param('~odor_threshold', 4.5)
        self.min_distance_between_points = 0.2  # Minimum distance to add a new point
        
        # Track trajectory and whiff points
        self.trajectory_points = []
        self.whiff_points = []
        self.last_pose = None
        
        # Marker IDs
        self.path_id = 0
        self.sphere_id = 100
        
        # For detecting whiffs
        self.current_odor = 0.0
        self.last_odor = 0.0
        self.hit_occurred = False
        
        # Publishers
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        
        # Subscribers
        self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.odor_sub = rospy.Subscriber('odor_reading', OdorReading, self.odor_callback)
        
        # Initialize visualization
        self.setup_visualization()
        
        rospy.loginfo("Gazebo visualizer initialized")
    
    def setup_visualization(self):
        """Initialize visualization markers"""
        # Create target marker (odor source)
        self.publish_sphere_marker(
            self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.3,  # Raise slightly above ground
            1.0, 0.5, 0.0, 1.0,  # Orange
            0.8,  # Larger size for visibility
            0,  # ID
            "target"
        )
        
        # Send initial empty path marker
        self.publish_path_markers()
    
    def pose_callback(self, msg):
        """Process drone position updates"""
        # Extract current position
        current_pos = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        
        # Add to trajectory if it's a new point
        if self.last_pose is None or self.distance(current_pos, self.last_pose) > self.min_distance_between_points:
            self.trajectory_points.append(current_pos)
            self.last_pose = current_pos
            
            # Update path visualization
            self.publish_path_markers()
    
    def odor_callback(self, msg):
        """Process odor readings"""
        self.current_odor = msg.concentration
        
        # Detect whiffs using the same logic as in the tracker
        if self.current_odor >= self.odor_threshold:
            if self.current_odor <= self.last_odor and not self.hit_occurred:
                # Whiff detected
                self.hit_occurred = True
                if self.last_pose is not None:
                    self.whiff_points.append(self.last_pose)
                    rospy.loginfo(f"Whiff detected at {self.last_pose}, concentration: {self.current_odor:.2f}")
                    
                    # Visualize whiff point
                    self.publish_sphere_marker(
                        self.last_pose[0], self.last_pose[1], self.last_pose[2] + 0.3,  # Raise above ground
                        1.0, 0.0, 0.0, 1.0,  # Red
                        0.5,  # Larger size for visibility
                        self.sphere_id,
                        "whiff"
                    )
                    self.sphere_id += 1
            
            self.last_odor = self.current_odor
        else:
            self.last_odor = 0
            self.hit_occurred = False
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    
    def publish_sphere_marker(self, x, y, z, r, g, b, a, scale, id, ns):
        """Publish a sphere marker"""
        marker = Marker()
        marker.header.frame_id = "world"  # Changed from "map" to "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        
        # Lifetime - persist for a long time
        marker.lifetime = rospy.Duration(0)  # 0 means forever
        
        # Add to marker array and publish
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
    
    def publish_path_markers(self):
        """Publish path as line strips"""
        if len(self.trajectory_points) < 2:
            return
        
        # Create marker array
        marker_array = MarkerArray()
        
        # Path marker - use spheres instead of line strip for better visibility
        path_marker = Marker()
        path_marker.header.frame_id = "world"  # Changed from "map" to "world"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "trajectory"
        path_marker.id = self.path_id
        path_marker.type = Marker.SPHERE_LIST
        path_marker.action = Marker.ADD
        
        # Sphere properties
        path_marker.scale.x = 0.2  # Sphere size
        path_marker.scale.y = 0.2
        path_marker.scale.z = 0.2
        path_marker.color.r = 0.0
        path_marker.color.g = 0.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0
        
        # Add points to path - raise them slightly above ground
        for point in self.trajectory_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2] + 0.3  # Raise path points above ground for visibility
            path_marker.points.append(p)
        
        marker_array.markers.append(path_marker)
        
        # Also add a line strip version for continuity
        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "trajectory_line"
        line_marker.id = self.path_id + 1000
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        
        line_marker.scale.x = 0.1  # Line width
        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        # Add points to line
        for point in self.trajectory_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2] + 0.3  # Raise line above ground
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Add start position marker - make it more visible
        if self.trajectory_points:
            start_marker = Marker()
            start_marker.header.frame_id = "world"
            start_marker.header.stamp = rospy.Time.now()
            start_marker.ns = "start"
            start_marker.id = 99
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            
            start_marker.pose.position.x = self.trajectory_points[0][0]
            start_marker.pose.position.y = self.trajectory_points[0][1]
            start_marker.pose.position.z = self.trajectory_points[0][2] + 0.3
            start_marker.pose.orientation.w = 1.0
            
            # Larger start marker
            start_marker.scale.x = 0.8
            start_marker.scale.y = 0.8
            start_marker.scale.z = 0.8
            
            start_marker.color.r = 0.0
            start_marker.color.g = 1.0
            start_marker.color.b = 0.0
            start_marker.color.a = 1.0
            
            marker_array.markers.append(start_marker)
        
        # Add current direction arrow - make it larger
        if len(self.trajectory_points) > 1:
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "world"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "direction"
            arrow_marker.id = 98
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # Arrow starts at current position
            start_point = Point()
            start_point.x = self.trajectory_points[-1][0]
            start_point.y = self.trajectory_points[-1][1]
            start_point.z = self.trajectory_points[-1][2] + 0.3
            
            # Calculate direction from last two points
            if len(self.trajectory_points) > 1:
                p1 = np.array(self.trajectory_points[-2])
                p2 = np.array(self.trajectory_points[-1])
                direction = p2 - p1
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    
                    # End point is 2m ahead in the direction of travel
                    end_point = Point()
                    end_point.x = start_point.x + direction[0] * 2.0
                    end_point.y = start_point.y + direction[1] * 2.0
                    end_point.z = start_point.z + direction[2] * 2.0
                    
                    arrow_marker.points.append(start_point)
                    arrow_marker.points.append(end_point)
                    
                    # Larger arrow
                    arrow_marker.scale.x = 0.2  # Shaft diameter
                    arrow_marker.scale.y = 0.4  # Head diameter
                    arrow_marker.scale.z = 0.6  # Head length
                    
                    arrow_marker.color.r = 0.0
                    arrow_marker.color.g = 0.0
                    arrow_marker.color.b = 1.0
                    arrow_marker.color.a = 1.0
                    
                    marker_array.markers.append(arrow_marker)
        
        # Set markers to persist
        for marker in marker_array.markers:
            marker.lifetime = rospy.Duration(0)  # 0 means forever
        
        # Publish all markers
        self.marker_pub.publish(marker_array)
        self.path_id += 1
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(10)  # 10 Hz update rate is sufficient for visualization
        
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == "__main__":
    try:
        visualizer = GazeboVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
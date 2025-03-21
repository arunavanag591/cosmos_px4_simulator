#!/usr/bin/env python
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from gazebo_px4_simulator.msg import OdorReading
from std_msgs.msg import ColorRGBA

class PlumeTrajVisualizer:
    def __init__(self):
        rospy.init_node('plume_trajectory_visualizer')
        
        # Parameters
        self.target_pos = rospy.get_param('~target_pos', [0.0, 0.0, 2.0])
        self.odor_threshold = rospy.get_param('~odor_threshold', 4.5)
        
        # Track points for trajectory
        self.trajectory_points = []
        self.whiff_points = []
        self.last_pose = None
        self.min_point_distance = 0.1  # Minimum distance between trajectory points
        
        # Create publishers for visualization markers
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('plume_tracking_path', Marker, queue_size=10)
        
        # Create subscribers
        self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.odor_sub = rospy.Subscriber('odor_reading', OdorReading, self.odor_callback)
        
        self.current_odor = 0.0
        self.last_odor = 0.0
        self.hit_occurred = False
        
        rospy.loginfo("Plume trajectory visualizer initialized")
    
    def pose_callback(self, msg):
        """Process drone position updates"""
        # Extract current position 
        current_pos = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        
        # Skip if too close to last point
        if self.last_pose is not None:
            dist = np.sqrt(sum([(a - b) ** 2 for a, b in zip(current_pos, self.last_pose)]))
            if dist < self.min_point_distance:
                return
        
        # Add to trajectory
        self.trajectory_points.append(current_pos)
        self.last_pose = current_pos
        
        # Publish visualization
        self.publish_visualization()
    
    def odor_callback(self, msg):
        """Process odor readings to detect whiffs"""
        # Store current odor reading
        self.current_odor = msg.concentration
        
        # Detect peaks using same logic as in tracker
        if self.current_odor >= self.odor_threshold:
            if self.current_odor <= self.last_odor and not self.hit_occurred:
                # Whiff detected
                self.hit_occurred = True
                # Add current position to whiff points
                if self.last_pose is not None:
                    self.whiff_points.append(self.last_pose)
                    rospy.loginfo(f"Whiff detected at {self.last_pose}, concentration: {self.current_odor:.2f}")
                    
                    # Update visualization immediately when a whiff is detected
                    self.publish_visualization()
            self.last_odor = self.current_odor
        else:
            self.last_odor = 0
            self.hit_occurred = False
    
    def publish_visualization(self):
        """Publish visualization markers for RViz"""
        marker_array = MarkerArray()
        
        # 1. Create path line marker
        if len(self.trajectory_points) > 1:
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = rospy.Time.now()
            path_marker.ns = "trajectory"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            
            path_marker.scale.x = 0.1  # Line width
            path_marker.color.r = 0.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0
            
            # Add points to path
            for point in self.trajectory_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                path_marker.points.append(p)
            
            marker_array.markers.append(path_marker)
            # Also publish as individual marker for subscribers that only listen to one topic
            self.path_pub.publish(path_marker)
        
        # 2. Create start point marker
        if self.trajectory_points:
            start_marker = Marker()
            start_marker.header.frame_id = "map"
            start_marker.header.stamp = rospy.Time.now()
            start_marker.ns = "start"
            start_marker.id = 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            
            start_marker.pose.position.x = self.trajectory_points[0][0]
            start_marker.pose.position.y = self.trajectory_points[0][1]
            start_marker.pose.position.z = self.trajectory_points[0][2]
            
            start_marker.pose.orientation.w = 1.0
            
            start_marker.scale.x = 0.5
            start_marker.scale.y = 0.5
            start_marker.scale.z = 0.5
            
            start_marker.color.r = 0.0
            start_marker.color.g = 1.0
            start_marker.color.b = 0.0
            start_marker.color.a = 1.0
            
            marker_array.markers.append(start_marker)
        
        # 3. Create target marker
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.header.stamp = rospy.Time.now()
        target_marker.ns = "target"
        target_marker.id = 2
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        
        target_marker.pose.position.x = self.target_pos[0]
        target_marker.pose.position.y = self.target_pos[1]
        target_marker.pose.position.z = self.target_pos[2]
        
        target_marker.pose.orientation.w = 1.0
        
        target_marker.scale.x = 0.5
        target_marker.scale.y = 0.5
        target_marker.scale.z = 0.5
        
        target_marker.color.r = 1.0
        target_marker.color.g = 0.5
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        
        marker_array.markers.append(target_marker)
        
        # 4. Create whiff markers
        for i, point in enumerate(self.whiff_points):
            whiff_marker = Marker()
            whiff_marker.header.frame_id = "map"
            whiff_marker.header.stamp = rospy.Time.now()
            whiff_marker.ns = "whiffs"
            whiff_marker.id = i + 100  # Offset to avoid ID conflicts
            whiff_marker.type = Marker.SPHERE
            whiff_marker.action = Marker.ADD
            
            whiff_marker.pose.position.x = point[0]
            whiff_marker.pose.position.y = point[1]
            whiff_marker.pose.position.z = point[2]
            
            whiff_marker.pose.orientation.w = 1.0
            
            whiff_marker.scale.x = 0.3
            whiff_marker.scale.y = 0.3
            whiff_marker.scale.z = 0.3
            
            whiff_marker.color.r = 1.0
            whiff_marker.color.g = 0.0
            whiff_marker.color.b = 0.0
            whiff_marker.color.a = 1.0
            
            marker_array.markers.append(whiff_marker)
        
        # 5. Create drone direction arrow
        if len(self.trajectory_points) > 1:
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "map"
            arrow_marker.header.stamp = rospy.Time.now()
            arrow_marker.ns = "direction"
            arrow_marker.id = 3
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # Arrow starts at current position
            start_point = Point()
            start_point.x = self.trajectory_points[-1][0]
            start_point.y = self.trajectory_points[-1][1]
            start_point.z = self.trajectory_points[-1][2]
            
            # Calculate direction from last two points
            if len(self.trajectory_points) > 1:
                p1 = np.array(self.trajectory_points[-2])
                p2 = np.array(self.trajectory_points[-1])
                direction = p2 - p1
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    
                    # End point is 1m ahead in the direction of travel
                    end_point = Point()
                    end_point.x = start_point.x + direction[0]
                    end_point.y = start_point.y + direction[1]
                    end_point.z = start_point.z + direction[2]
                    
                    arrow_marker.points.append(start_point)
                    arrow_marker.points.append(end_point)
                    
                    arrow_marker.scale.x = 0.1  # Shaft diameter
                    arrow_marker.scale.y = 0.2  # Head diameter
                    arrow_marker.scale.z = 0.3  # Head length
                    
                    arrow_marker.color.r = 0.0
                    arrow_marker.color.g = 0.0
                    arrow_marker.color.b = 1.0
                    arrow_marker.color.a = 1.0
                    
                    marker_array.markers.append(arrow_marker)
        
        # Publish all markers
        self.marker_pub.publish(marker_array)
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz update rate is sufficient for visualization
        
        while not rospy.is_shutdown():
            # Visualization is triggered by callbacks
            rate.sleep()

if __name__ == "__main__":
    try:
        visualizer = PlumeTrajVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
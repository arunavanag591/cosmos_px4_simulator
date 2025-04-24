# COSMOS Integrated Gazebo simulation for odor tracking and testing
Odor simulator for gazebo-classic and ROS 1.0 using PX4 controller


Installation procedure:

1. [Setup Gazebo and PX4](https://docs.px4.io/main/en/sim_gazebo_classic/)
2. [Setup PX4 Firmware](https://github.com/PX4/PX4-Autopilot)
3. [Setup your workspace](https://docs.px4.io/main/en/ros/mavros_installation.html)
4. You might need to install the following dependencies: 

    ```bash
    sudo apt-get install libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly -y
    ``` 
5. Add the following paths in your `.bashrc`
    ``` bash
    source ~/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic
    export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-9/plugins
    ```

6. How to use?:
    - To run it with CSV
        ``` bash
        cd ~/PX4-Autopilot/launch/
        roslaunch mavros_posix_sitl.launch
        rosrun gazebo_px4_simulator csv_trajectory.py
        
    - Velocity control with cosmos odor simulator
        ``` bash
        cd ~/PX4-Autopilot/launch/
        roslaunch mavros_posix_sitl.launch
        rosrun gazebo_px4_simulator surge_cast_vel.py x y (e.g. 40 -8)
        ```
    - Position control with cosmos odor simulator
        ``` bash
        cd ~/PX4-Autopilot/launch/
        roslaunch mavros_posix_sitl.launch
        rosrun gazebo_px4_simulator surge_cast_position.py x y (e.g. 40 -8)
        ```
    - To visualize in rviz:
        ``` bash
        roslaunch gazebo_px4_simulator tracking_rviz_visualization.launch 
        ```

    - Edited world file is [here](/odor_sim_assets/world/mcmillan_airfield.world), that can be replaced with the system world file, usually found in the below directory:
        ``` bash
        cd ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/ 
        ```

    - To add wind forces to the gazebo-classic environment, add the below lines to the world file 

        ```
        <plugin name="WindPublisherPlugin" filename="libWindPublisherPlugin.so">
    	<frameId>world</frameId>
  		<windVelocityMean>6.0</windVelocityMean>
  		<windVelocityMax>8.0</windVelocityMax>
  		<windVelocityVariance>1.0</windVelocityVariance>
  		<windDirectionMean>1 0 0</windDirectionMean>
  		<windGustStart>1.0</windGustStart>
  		<windGustDuration>2.0</windGustDuration>
  		<gustVelocityMean>8.0</gustVelocityMean>
  		<gustVelocityVariance>2.0</gustVelocityVariance>
		</plugin>
		<sensor name="wind_sensor" type="custom">
  	    <always_on>1</always_on>
  	    <update_rate>10</update_rate>
  	    <plugin name="wind_sensor_plugin" filename="libWindSensorPlugin.so">
  	    <topic>/drone/wind</topic>
  	    </plugin>
		</sensor>
        ```
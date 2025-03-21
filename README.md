# Gazebo PX4 Simulator
PX4 simulator for gazebo-classic and ROS 1.0


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
        rosrun gazebo_px4_simulator gazebo_px4_simulator.py
        
    - To run it with cosmos odor simulator
        ``` bash
        cd ~/PX4-Autopilot/launch/
        roslaunch mavros_posix_sitl.launch
        rosrun gazebo_px4_simulator offboard_px4_simulator.py
        rosrun gazebo_px4_simulator tracking_controller_node.py
        ```
    - To visualize in rviz:
        ``` bash
        rosrun rviz rviz -d ~/gazebo_ws/src/gazebo_px4_simulator/launch/plume_tracking.rviz

        rosrun gazebo_px4_simulator plume_viz_node.py _target_pos:="[0.0, 0.0, 2.0]" _odor_threshold:="4.5"
        ```

    - For checking the wind plugin go to:
        ``` bash
        ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world
        ```

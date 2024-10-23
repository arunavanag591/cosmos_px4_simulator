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


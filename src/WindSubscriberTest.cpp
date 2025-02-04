#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include "Wind.pb.h"
#include <iostream>

// Callback to handle wind messages
void OnWindMsg(const std::string &msg) {
  // Deserialize the message
  physics_msgs::msgs::Wind windMsg;
  if (!windMsg.ParseFromString(msg)) {
    std::cerr << "Failed to parse wind message" << std::endl;
    return;
  }

  // Output the wind data
  std::cout << "Received Wind Message:\n";
  std::cout << "Frame ID: " << windMsg.frame_id() << "\n";
  std::cout << "Time (usec): " << windMsg.time_usec() << "\n";
  std::cout << "Velocity: (" << windMsg.velocity().x() << ", "
            << windMsg.velocity().y() << ", " << windMsg.velocity().z() << ")\n";
}

int main(int argc, char **argv) {
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();

    // Subscribe to the topic
    gazebo::transport::SubscriberPtr sub =
        node->Subscribe("/gazebo/default/world_wind", OnWindMsg);

    // Keep the program running
    while (true) {
        gazebo::common::Time::MSleep(100);
    }

    return 0;
}

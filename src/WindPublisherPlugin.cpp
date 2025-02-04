#include "gazebo_px4_simulator/WindPublisherPlugin.hh"
#include "Wind.pb.h"

namespace gazebo {

WindPublisherPlugin::~WindPublisherPlugin() {}

void WindPublisherPlugin::Load(physics::WorldPtr world, sdf::ElementPtr sdf) {
  this->world=world;
  this->gzNode = gazebo::transport::NodePtr(new gazebo::transport::Node());
  this->gzNode->Init();
  this->windPub = this->gzNode->Advertise<physics_msgs::msgs::Wind>("~/world_wind");

  // Initialize random distributions
  windStrengthDistribution = std::normal_distribution<double>(this->windVelocityMean, sqrt(this->windVelocityVariance));
  gustStrengthDistribution = std::normal_distribution<double>(this->gustVelocityMean, sqrt(this->gustVelocityVariance));

  if (sdf->HasElement("frameId")) {
    this->frameId = sdf->Get<std::string>("frameId");
  }
  // Parse parameters from SDF
  if (sdf->HasElement("windVelocityMean")) {
    this->windVelocityMean = sdf->Get<double>("windVelocityMean");
  }
  if (sdf->HasElement("windVelocityVariance")) {
    this->windVelocityVariance = sdf->Get<double>("windVelocityVariance");
  }
  if (sdf->HasElement("windDirectionMean")) {
    this->windDirectionMean = sdf->Get<ignition::math::Vector3d>("windDirectionMean");
    this->windDirectionMean.Normalize();
  }
  if (sdf->HasElement("gustStart")) {
    this->gustStart = sdf->Get<double>("gustStart");
  }
  if (sdf->HasElement("gustDuration")) {
    this->gustDuration = sdf->Get<double>("gustDuration");
  }
  if (sdf->HasElement("gustVelocityMean")) {
    this->gustVelocityMean = sdf->Get<double>("gustVelocityMean");
  }
  if (sdf->HasElement("gustVelocityVariance")) {
    this->gustVelocityVariance = sdf->Get<double>("gustVelocityVariance");
  }

  this->lastUpdateTime = world->SimTime();

  // Connect to Gazebo's update event
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&WindPublisherPlugin::OnUpdate, this));bcampus.unr.edu/
}

void WindPublisherPlugin::OnUpdate() {
  common::Time now = this->world->SimTime();

  // Ensure publishing interval is respected
  if ((now - this->lastUpdateTime).Double() < this->publishInterval) {
    return;
  }
  this->lastUpdateTime = now;

  // Simulate steady wind
  double windStrength = std::max(0.0, windStrengthDistribution(generator));
  ignition::math::Vector3d windDirection = windDirectionMean;
  windDirection.Normalize();

  // Create wind message
  physics_msgs::msgs::Wind windMsg;

  // Set velocity
  windMsg.mutable_velocity()->set_x(windStrength * windDirection.X());
  windMsg.mutable_velocity()->set_y(windStrength * windDirection.Y());
  windMsg.mutable_velocity()->set_z(windStrength * windDirection.Z());

  // Set required fields
  windMsg.set_frame_id(this->frameId);                          // Set the frame ID
  windMsg.set_time_usec(this->world->SimTime().Double() * 1e6); // Set the current time in microseconds

  // Publish wind data
  windPub->Publish(windMsg);
}

// Register the plugin with Gazebo
GZ_REGISTER_WORLD_PLUGIN(WindPublisherPlugin)

}  // namespace gazebo

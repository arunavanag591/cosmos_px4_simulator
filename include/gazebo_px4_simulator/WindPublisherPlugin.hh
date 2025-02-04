#pragma once

#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>
#include "Wind.pb.h"
#include <random>

namespace gazebo {

class WindPublisherPlugin : public WorldPlugin {
 public:
  WindPublisherPlugin() = default;
  ~WindPublisherPlugin() override;
  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override;

 private:
  void OnUpdate();

  // Gazebo-specific transport
  gazebo::transport::NodePtr gzNode;
  gazebo::transport::PublisherPtr windPub;
  std::string frameId = "world";  // Default frame ID

  // Wind parameters
  double windVelocityMean = 10.0;
  double windVelocityVariance = 1.0;
  ignition::math::Vector3d windDirectionMean = ignition::math::Vector3d(1.0, 0.0, 0.0);

  // Gust parameters
  double gustStart = 5.0;
  double gustDuration = 2.0;
  double gustVelocityMean = 15.0;
  double gustVelocityVariance = 2.0;

  // Random number distributions
  std::default_random_engine generator;
  std::normal_distribution<double> windStrengthDistribution;
  std::normal_distribution<double> gustStrengthDistribution;

  common::Time lastUpdateTime;
  double publishInterval = 0.1;  // Publish at 10 Hz

  event::ConnectionPtr updateConnection;
  physics::WorldPtr world;
};

}  // namespace gazebo

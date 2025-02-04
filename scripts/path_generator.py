import csv
import numpy as np
import matplotlib.pyplot as plt

# Configuration
amplitude = 2  # meters
frequency = 1  # Hz (adjusted to ensure the sinusoid stays within 5 meters)
num_points = 100
step_size = 0.05  # meters between points, ensuring the total path length is 5 meters

# Generate sinusoidal path
x_values = np.linspace(0, 5, num_points)  # Restrict the path to 5 meters in total
y_values = amplitude * np.sin(2 * np.pi * frequency * x_values)
z_values = 2 * np.ones(num_points)  # Constant altitude
yaw_values = np.unwrap(np.arctan2(np.gradient(y_values), np.gradient(x_values)))  # Yaw facing direction of movement

# Write trajectory to CSV
output_file = '../trajectories/sin_trajectory.csv'
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(num_points):
        csvwriter.writerow([x_values[i], y_values[i], z_values[i], yaw_values[i]])

print(f"Trajectory written to {output_file}")

# Visualize the generated sinusoidal path
plt.figure()
plt.plot(x_values, y_values, label='Sinusoidal Path')
plt.quiver(x_values, y_values, np.cos(yaw_values), np.sin(yaw_values), scale=20, width=0.003, color='r', label='Yaw Direction')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Generated Sinusoidal Path with Yaw Direction')
plt.legend()
plt.grid()
plt.show()


import numpy as np
import pandas as pd

class SurgeCastAgent:
    def __init__(self, tau=0.42, noise=1.9, bias=0.25, threshold=4.5,
                 hit_trigger='peak', surge_amp=2.0, tau_surge=0.5,
                 cast_freq=1.0, cast_width=0.8, bounds=None):
        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.threshold = threshold
        self.hit_trigger = hit_trigger
        self.surge_amp = surge_amp
        self.tau_surge = tau_surge
        self.cast_freq = cast_freq
        self.cast_width = cast_width
        self.bounds = bounds
        self.surge_amp_ = surge_amp / (tau_surge * np.exp(-1))

    def reflect_if_out_of_bounds(self, v: np.ndarray, x: np.ndarray):
        if self.bounds is None:
            return v, x
        v_new = v.copy()
        x_new = x.copy()
        for dim in range(2):
            if x[dim] < self.bounds[dim][0]:
                v_new[dim] *= -1
                x_new[dim] = 2*self.bounds[dim][0] - x[dim]
            elif x[dim] > self.bounds[dim][1]:
                v_new[dim] *= -1
                x_new[dim] = 2*self.bounds[dim][1] - x[dim]
        return v_new, x_new

def tracking(predictor, bounds, start_pos, target_pos, surge_agent, 
          target_weight, plume_timeout, closest_to_source, duration):
    dt = 0.005
    n_steps = int(duration / dt)

    # Original arrays
    ts = np.arange(n_steps)*dt
    odors = np.zeros(n_steps)
    surges = np.zeros(n_steps)
    bs = np.zeros((n_steps,2))
    vs = np.zeros((n_steps,2))
    xs = np.zeros((n_steps,2))
    hits = np.zeros(n_steps)
    
    # New arrays for additional metrics
    velocities = np.zeros((n_steps,2))  # Velocity vector
    speeds = np.zeros(n_steps)          # Speed magnitude
    accelerations = np.zeros(n_steps)   # Acceleration magnitude
    angles = np.zeros(n_steps)          # Heading angle
    angular_velocities = np.zeros(n_steps)  # Rate of turning
    crosswind_distances = np.zeros(n_steps) # Distance perpendicular to wind
    upwind_distances = np.zeros(n_steps)    # Distance parallel to wind
    dist_to_targets = np.zeros(n_steps)     # Distance to target
    time_since_last_hits = np.zeros(n_steps) # Time since last whiff
    casting_phases = np.zeros(n_steps)      # Phase of casting motion
    local_curvatures = np.zeros(n_steps)    # Path curvature

    x = start_pos.copy()
    v = np.zeros(2)
    last_hit_time = -np.inf
    last_odor = 0
    hit_occurred = False
    prev_angle = 0

    # Initial bias calculation remains same
    to_target = target_pos - x
    to_target /= (np.linalg.norm(to_target) + 1e-6)
    upwind = np.array([-1.0, 0.0])
    b = (1 - target_weight)*upwind + target_weight*to_target
    b *= (surge_agent.bias / np.linalg.norm(b))

    for t_ctr in range(n_steps):
        current_odor = predictor.step_update(x[0], x[1], dt)
        odors[t_ctr] = current_odor

        # Original hit detection logic
        if surge_agent.hit_trigger == 'peak':
            if current_odor >= surge_agent.threshold:
                if current_odor <= last_odor and not hit_occurred:
                    hits[t_ctr] = 1
                    hit_occurred = True
                    last_hit_time = t_ctr*dt
                    remaining_steps = n_steps - t_ctr
                    ts_ = ts[:remaining_steps] - ts[0]
                    surge_force = surge_agent.surge_amp_ * ts_ * np.exp(-ts_/surge_agent.tau_surge)
                    surges[t_ctr:] = np.minimum(surge_force, 50.0)
                last_odor = current_odor
            else:
                last_odor = 0
                hit_occurred = False

        if t_ctr > 0:
            # Original movement logic
            eta = np.random.normal(0, surge_agent.noise, 2)
            time_since_hit = (t_ctr*dt - last_hit_time)
            to_target = target_pos - x
            dist_to_target = np.linalg.norm(to_target)
            
            if dist_to_target < closest_to_source:
                print(f"Target reached at {x}")
                break
                
            to_target /= (dist_to_target + 1e-6)

            current_target_weight = target_weight
            if time_since_hit > plume_timeout:
                current_target_weight = min(0.8, 
                    target_weight + 0.1*(time_since_hit - plume_timeout)/plume_timeout)

            if surges[t_ctr] > 1.0:
                surge_direction = np.array([-1.0, -0.05*x[1]])
                surge_direction /= np.linalg.norm(surge_direction)
                b = (1 - current_target_weight)*surge_direction + current_target_weight*to_target
                b *= surges[t_ctr]
            else:
                cast_freq = 0.5
                cast_phase = np.sin(2*np.pi*cast_freq*ts[t_ctr])
                base_cast_width = 1.0
                dist_factor = min(1.0, dist_to_target/20.0)
                cast_width = base_cast_width*dist_factor

                crosswind = np.array([0.0, cast_phase*cast_width])
                upwind = np.array([-0.2, 0.0])
                b = (1 - current_target_weight)*(upwind + crosswind) + current_target_weight*to_target
                norm_b = np.linalg.norm(b)
                if norm_b > 0:
                    b *= surge_agent.bias/norm_b

            # Update velocity and position
            v += (dt/surge_agent.tau)*(-v + eta + b)
            x += v*dt
            v, x = surge_agent.reflect_if_out_of_bounds(v, x)

            # Calculate additional metrics
            velocities[t_ctr] = v
            speeds[t_ctr] = np.linalg.norm(v)
            if t_ctr > 0:
                accelerations[t_ctr] = (speeds[t_ctr] - speeds[t_ctr-1])/dt
            
            # Heading angle and angular velocity
            current_angle = np.arctan2(v[1], v[0])
            angles[t_ctr] = current_angle
            if t_ctr > 0:
                angle_diff = np.arctan2(np.sin(current_angle - prev_angle),
                                      np.cos(current_angle - prev_angle))
                angular_velocities[t_ctr] = angle_diff/dt
            prev_angle = current_angle
            
            # Distance components
            crosswind_distances[t_ctr] = abs(x[1] - target_pos[1])
            upwind_distances[t_ctr] = abs(x[0] - target_pos[0])
            dist_to_targets[t_ctr] = dist_to_target
            time_since_last_hits[t_ctr] = time_since_hit
            casting_phases[t_ctr] = cast_phase
            
            # Path curvature (for segments of 3 points)
            if t_ctr >= 2:
                pos_window = xs[t_ctr-2:t_ctr+1]
                dx = np.gradient(pos_window[:,0])
                dy = np.gradient(pos_window[:,1])
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
                local_curvatures[t_ctr] = np.mean(curvature)

        # Store original metrics
        bs[t_ctr] = b
        vs[t_ctr] = v
        xs[t_ctr] = x

    # Trim if ended early
    if t_ctr < (n_steps-1):
        trim_slice = slice(0, t_ctr+1)
        ts = ts[trim_slice]
        xs = xs[trim_slice]
        bs = bs[trim_slice]
        vs = vs[trim_slice]
        odors = odors[trim_slice]
        hits = hits[trim_slice]
        surges = surges[trim_slice]
        velocities = velocities[trim_slice]
        speeds = speeds[trim_slice]
        accelerations = accelerations[trim_slice]
        angles = angles[trim_slice]
        angular_velocities = angular_velocities[trim_slice]
        crosswind_distances = crosswind_distances[trim_slice]
        upwind_distances = upwind_distances[trim_slice]
        dist_to_targets = dist_to_targets[trim_slice]
        time_since_last_hits = time_since_last_hits[trim_slice]
        casting_phases = casting_phases[trim_slice]
        local_curvatures = local_curvatures[trim_slice]

    # Create complete DataFrame
    trajectory_df = pd.DataFrame({
        'time': ts,
        'x': xs[:,0],
        'y': xs[:,1],
        'vx': velocities[:,0],
        'vy': velocities[:,1],
        'speed': speeds,
        'acceleration': accelerations,
        'heading_angle': angles,
        'angular_velocity': angular_velocities,
        'crosswind_dist': crosswind_distances,
        'upwind_dist': upwind_distances,
        'dist_to_target': dist_to_targets,
        'time_since_whiff': time_since_last_hits,
        'casting_phase': casting_phases,
        'path_curvature': local_curvatures,
        'odor': odors,
        'whiff': hits,
        'surge_force': surges,
        'bias_force_x': bs[:,0],
        'bias_force_y': bs[:,1]
    })
    

    if trajectory_df.iloc[-1]['x'] == 0 and trajectory_df.iloc[-1]['y'] == 0:
            trajectory_df = trajectory_df.iloc[:-1]
    return trajectory_df
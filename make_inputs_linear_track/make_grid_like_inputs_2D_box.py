#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
2D Grid Field Input Generation
Includes overlap-based freezing logic: Movement velocity is 
dynamically linked to the activation of familiar place fields (CFC engram).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

# 7x7 grid of place fields = 49 fields (evenly spaced across arena)
import numpy as np
num_fields_per_dim = 7
x_array = np.linspace(0, 100, num_fields_per_dim).astype(int)
y_array = np.linspace(0, 100, num_fields_per_dim).astype(int)

# Arena size (2D box boundaries)
myx = 100
myy = 100

my_run = int(sys.argv[1]) if len(sys.argv) > 1 else 0
my_cond = int(sys.argv[2]) if len(sys.argv) > 2 else 0
print("Run: " + str(my_run) + ", Condition: " + str(my_cond))

maindir1 = 'runs_produced_2D_cond' + str(my_cond) + '_pos/'
maindir2 = 'runs_produced_2D_cond' + str(my_cond) + '_neg/'
dirname1 = maindir1 + 'run_' + str(my_run)
dirname2 = maindir2 + 'run_' + str(my_run)

# ---- CACHE CHECK ----
# If inputs already exist for this run/condition, skip regeneration
missing_files = False

# 1. Check for trajectory files
if not os.path.exists(dirname1 + '/path.txt') or not os.path.exists(dirname1 + '/velocity.txt'):
    missing_files = True

# 2. Check for the VERY LAST spike file expected
ndend = 8

total_fields = len(x_array) * len(y_array)
last_idx = (total_fields - 1) * ndend + (ndend - 1)
last_expected_file = dirname2 + '/g' + str(last_idx) + '_CA3.txt'
last_expected_MnR_file = dirname1 + '/g' + str(last_idx) + '_MnR.txt'

if not os.path.exists(last_expected_file) or not os.path.exists(last_expected_MnR_file):
    missing_files = True

if not missing_files:
    print('All inputs (Path, Vel, and 392 Spike files) exist for cond ' + str(my_cond) + ' run ' + str(my_run) + '.')
    print('Skipping generation.')
    sys.exit(0)
else:
    print('Cache incomplete or missing for Run ' + str(my_run) + '. Starting generation...')

os.system('mkdir -p ' + dirname1)
os.system('mkdir -p ' + dirname2)

# Ensure reproducibility for this specific run
np.random.seed(my_run)

# Baseline place field mapping (matching the CFC context)
baseline_locations = [(x, y) for x in x_array for y in y_array]
current_locations = list(baseline_locations)

# Remap place fields for the Neutral Context
if my_cond == 3:
    np.random.seed(my_run + 3) # Consistent remapping map
    np.random.shuffle(current_locations)
    np.random.seed(my_run) # Restore seed for random walk

# Simulation duration
n_ticks = 22500 

# --- Realistic Velocity Parameters (cm/ms) ---
# Mouse speed 10-20 cm/s -> 0.01 to 0.02 cm/ms
V_MOVE = 0.025 #0.035          # Standard "Exploration" speed
V_SURPRISE = 0.080       # During foot shocks speed increases
FRICTION = 0.99        # Velocity decay per ms (0.85-0.95 range)
ACCEL_KICK = 0.1     # Random acceleration push
MOMENTUM = 0.99         # How much of the previous heading is kept (0.0 to 1.0)
angle = np.random.rand() * 2 * np.pi  # Initial random direction

# State-switching probabilities (per ms)
# 0.001 prob means the state lasts ~1000ms (1 second) on average
prob_to_pause_init = 0.0005
prob_to_move_init = 0.0024 # 0.0032

is_moving = True
speed_state_proportion = 0.7 # higher ratio for slower speeds
if np.random.random() < speed_state_proportion:
    # Exponential decay: most values near 0, mean of 0.2
    speed_factor = np.random.exponential(0.1)
else:
    # Gaussian "hump" centered near the max speed (0.85) with some spread (0.1)
    speed_factor = np.random.normal(0.75, 0.25)
scaled_v_move = V_MOVE * np.min([speed_factor, 1.0])

pos = np.array([50.0, 50.0])
vel = np.array([0.0, 0.0])

# Random Walk Trajectory
path = np.zeros((n_ticks, 2))
velocity_record = np.zeros(n_ticks)

# Place field variables
place_field_overlap = 1.5 # 1.15=15% overlap between fields

if my_cond == 2:
    context_fear_level = 0.5  # Full fear in Reintroduction
elif my_cond == 3:
    context_fear_level = 0.2  # Reduced "Generalization" fear in Neutral
else:
    context_fear_level = 0.0  # No fear in Novelty

# def get_overlap_ratio(p_x, p_y):
#     # Calculate how many of the currently active place fields match their CFC locations
#     total_active = 0.0
#     fear_active = 0.0

#     sigma_sens = place_field_overlap * myx / (len(x_array)-1)
#     for k in range(len(current_locations)):
#         cx, cy = current_locations[k]
#         dist_sq = (p_x - cx)**2 + (p_y - cy)**2
#         act = np.exp(-dist_sq / (2 * sigma_sens**2))
#         if act > 0.1: # Only consider fields that are reasonably active
#             total_active += act
#             if current_locations[k] == baseline_locations[k]:
#                 fear_active += act
    
#     return fear_active / total_active if total_active > 0 else 0.0

visit_res = 20  # 100cm / 5cm = 20 bins
visitation_map = np.zeros((visit_res, visit_res))
shock_record = np.zeros(n_ticks, dtype=bool)

times_arr = np.arange(n_ticks)
novelty_boost_arr = np.zeros(n_ticks) # global novelty
if my_cond == 0 or my_cond == 3 or my_cond == 4:
    if my_cond == 0 or my_cond == 3: # starts novel immediately
        familiar_period = 0
    elif my_cond == 4: # starts novel midway
        familiar_period = 11500
    novelty_boost_arr = np.where(times_arr < familiar_period, 0, 1)

for i in range(n_ticks):
    is_shock = False
    
    if my_cond == 1:
        # Fear conditioning occurs progressively during CFC
        if i == 5500 + 1500: context_fear_level = 0.2
        if i == 8500 + 1500: context_fear_level = 0.3
        if i == 11500 + 1500: context_fear_level = 0.4
        if i == 14500 + 1500: context_fear_level = 0.5
        
        # Immediate shock freezing
        for shock_t in [5500, 8500, 11500, 14500]:
            if i >= shock_t and i < shock_t + 1500:
                is_shock = True

    if is_shock: 
        shock_record[i] = True
        is_moving = True

    global_novelty_boost = 1.0 # max 100% boost in move probability
    novelty_move_prob = global_novelty_boost * prob_to_move_init * novelty_boost_arr[i]

    prob_to_pause = prob_to_pause_init + prob_to_pause_init * context_fear_level # increase prob to pause with fear
    prob_to_move = prob_to_move_init - prob_to_move_init * context_fear_level + novelty_move_prob # decrease prob to move with fear

    # State Switching (Move vs. Pause)
    rnd_state = np.random.rand()
    if is_moving and rnd_state < prob_to_pause:
        is_moving = False
    elif not is_moving and rnd_state < prob_to_move:
        is_moving = True
        if np.random.random() < speed_state_proportion:
            # Exponential decay: most values near 0, mean of 0.2
            speed_factor = np.random.exponential(0.1)
        else:
            # Gaussian "hump" centered near the max speed (0.85) with some spread (0.1)
            speed_factor = np.random.normal(0.75, 0.25)
        scaled_v_move = V_MOVE * np.min([speed_factor, 1.0]) # sets max speed for this epoch
    
    novelty_speed_prob = global_novelty_boost * scaled_v_move * 1.5 * novelty_boost_arr[i]

    # Calculate exactly how much the current location represents the CFC fear context
    # overlap_ratio = get_overlap_ratio(pos[0], pos[1])
    
    # max_vel is determined by baseline speed (V_MOVE) modified by fear/shocks
    if is_shock:
        max_vel = V_SURPRISE # increased speed during shock
    else:
        # Scale speed based on fear level
        max_vel = scaled_v_move - scaled_v_move * context_fear_level + novelty_speed_prob # * overlap_ratio

    # If the state is 'paused', limit speed even further
    if not is_moving:
        max_vel = 0.0001

    # 4. Physics Update: Friction + Acceleration
    if is_moving:
        # 1. Update map (use a smaller increment to keep gradients sensitive)
        map_x = int(np.clip(pos[0] / 5.0, 0, visit_res - 1))
        map_y = int(np.clip(pos[1] / 5.0, 0, visit_res - 1))
        visitation_map[map_x, map_y] += 0.01

        # 2. Add random jitter (The "Whim" factor)
        # Increasing this from 0.1 to 0.15 or 0.2 will break the straight lines
        angle += np.random.randn() * 0.15

        # 3. Novelty Steering
        counts_up    = visitation_map[map_x, min(map_y+1, visit_res-1)]
        counts_down  = visitation_map[map_x, max(map_y-1, 0)]
        counts_left  = visitation_map[max(map_x-1, 0), map_y]
        counts_right = visitation_map[min(map_x+1, visit_res-1), map_y]

        dx = counts_left - counts_right
        dy = counts_down - counts_up
        
        target_angle_novelty = np.arctan2(dy, dx)
        angle_diff = (target_angle_novelty - angle + np.pi) % (2 * np.pi) - np.pi
        
        # LOWER this weight (0.01 instead of 0.05) to let randomness play a bigger role
        angle += angle_diff * 0.01 
        
        # 4. Momentum Physics
        target_vel = np.array([np.cos(angle), np.sin(angle)]) * max_vel
        vel = (vel * MOMENTUM) + (target_vel * (1.0 - MOMENTUM))
    else:
        # Natural decay when pausing
        vel *= FRICTION
    
    # Update velocity
    speed = np.linalg.norm(vel)
    if speed > max_vel:
        vel = (vel / speed) * max_vel
        speed = max_vel
    
    velocity_record[i] = speed

    pos += vel
    
    # Boundary conditions
    if pos[0] < 1: pos[0] = 1; vel[0] *= -1
    if pos[0] > myx-1: pos[0] = myx-1; vel[0] *= -1
    if pos[1] < 1: pos[1] = 1; vel[1] *= -1
    if pos[1] > myy-1: pos[1] = myy-1; vel[1] *= -1
    
    path[i] = pos
    
    # Record location novelty based on how little this area has been explored
    curr_map_x = int(np.clip(pos[0] / 5.0, 0, visit_res - 1))
    curr_map_y = int(np.clip(pos[1] / 5.0, 0, visit_res - 1))

np.savetxt(dirname1 + '/path.txt', path, fmt='%.1f', delimiter=' ')
np.savetxt(dirname1 + '/velocity.txt', velocity_record, fmt='%.4f')
print('Done with the 2D path for condition ' + str(my_cond))
print("MAX SPEED ACHIEVED: %.2f cm/s" % (np.max(velocity_record) * 1000))
print("TOTAL DISTANCE: %.1f cm" % np.sum(velocity_record))
print("FINAL POSITION: ", path[-1])

# Plot speed over time
plot_dir = dirname1 + '/Probability_Maps/'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)

fig_vel, ax_vel = plt.subplots(figsize=(10, 4))
time_s = np.arange(n_ticks) / 1000.0
speed_cms = velocity_record * 1000.0
ax_vel.plot(time_s, speed_cms, color='black', linewidth=1.5)
ax_vel.set_xlabel('Time (s)')
ax_vel.set_ylabel('Speed (cm/s)')
ax_vel.set_title('Speed vs Time (Condition %d)' % my_cond)
fig_vel.tight_layout()
fig_vel.savefig(plot_dir + 'Speed_vs_Time.png', dpi=150)
plt.close(fig_vel)

# Calculate locomotion metrics
thresh_cm_s = 2.0
is_loco = speed_cms > thresh_cm_s
diffs = np.diff(is_loco.astype(int))
starts = list(np.where(diffs == 1)[0] + 1)
if is_loco[0]:
    starts.insert(0, 0)
ends = list(np.where(diffs == -1)[0] + 1)
if len(is_loco) > 0 and is_loco[-1]:
    ends.append(len(is_loco))

starts = np.array(starts)
ends = np.array(ends)

num_epochs = len(starts)
total_time_s = n_ticks / 1000.0
epochs_per_s = num_epochs / total_time_s

durations_s = (ends - starts) / 1000.0
avg_duration_s = np.mean(durations_s) if num_epochs > 0 else 0.0

mean_loco_speed = np.mean(speed_cms[is_loco]) if np.any(is_loco) else 0.0
mean_speed_overall = np.mean(speed_cms)
max_speed = np.max(speed_cms)

# Save metrics to CSV
csv_path = plot_dir + 'Speed_Metrics.csv'
with open(csv_path, 'w') as f:
    f.write("Num_Movement_Epochs_per_s,Avg_Movement_Epoch_Duration_s,Mean_Speed_locomotion_cm_s,Mean_Speed_overall_cm_s,Max_Speed_cm_s\n")
    f.write("%f,%f,%f,%f,%f\n" % (epochs_per_s, avg_duration_s, mean_loco_speed, mean_speed_overall, max_speed))

# Grid Like Inputs Generation
theta_freq = 8
theta_phase = 0

def vectorized_gridfield(theta, lambda_var, xo, yo, x, y):
    th1_0, th1_1 = np.cos(theta), np.sin(theta)
    th2_0, th2_1 = np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)
    th3_0, th3_1 = np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)
    dx = x - xo
    dy = y - yo
    k = (4 * np.pi) / (np.sqrt(3) * lambda_var)
    dot1 = dx * th1_0 + dy * th1_1
    dot2 = dx * th2_0 + dy * th2_1
    dot3 = dx * th3_0 + dy * th3_1
    return (1/4.5) * (np.cos(k * dot1) + np.cos(k * dot2) + np.cos(k * dot3) + 1.5)

theta_mod = (np.sin(2.0*np.pi*theta_freq * times_arr/1000.0 + theta_phase)+1.0)/2.0
norm_speed = velocity_record / V_MOVE
norm_speed = np.clip(norm_speed, 0, 1) # ensures speed factor stays between 0 and 1, since V_SURPRISE is higher than V_MOVE

p_pos_EC_factor = (0.05 + 0.95 * norm_speed) * (0.8 + 0.2 * novelty_boost_arr) # previously 0.7 + 0.3 * norm_speed
p_pos_CA3_factor = (0.7 + 0.3 * norm_speed) # previously 0.7 + 0.3 * norm_speed
p_neg_EC_factor = (1.0 - 0.95 * norm_speed) * (0.8 + 0.2 * novelty_boost_arr) # prveviously 1.0 - 0.95 * norm_speed
p_neg_CA3_factor = (1.0 - 0.3 * norm_speed) # previously 1.0 - 0.95 * norm_speed

regular_spatial_thresh_EC = 0.7
regular_spatial_thresh_CA3 = 0.5
shock_spatial_thresh_EC = regular_spatial_thresh_EC-0.15
shock_spatial_thresh_CA3 = regular_spatial_thresh_CA3-0.15

spatial_thresh_EC_arr = np.where(shock_record, shock_spatial_thresh_EC, regular_spatial_thresh_EC)
spatial_thresh_CA3_arr = np.where(shock_record, shock_spatial_thresh_CA3, regular_spatial_thresh_CA3)

for k in range(len(current_locations)):
    xxx, yyy = current_locations[k]
    my_field = k + 1

    fig_g, ax_g = plt.subplots(1, 1, figsize=(7, 7))
    ax_g.set_title('Average EC/CA3:\nField %d (Center: %d, %d)' % (my_field, xxx, yyy), fontsize=22, fontweight='bold')
    fig, axes = plt.subplots(4, 4, figsize=(18, 20))
    fig.suptitle('EC/CA3 Probability Maps: Field %d (Center: %d, %d)' % (my_field, xxx, yyy), fontsize=22, fontweight='bold')
    
    res = 100
    arena_x = np.linspace(0, myx, res)
    arena_y = np.linspace(0, myy, res)
    X, Y = np.meshgrid(arena_x, arena_y)
    
    grid_orientation = 0.0
    target_lambda = place_field_overlap * myx / (len(x_array) - 1)
    field_increment = (target_lambda / 2.0) / (ndend - 1)
    lambda_base = target_lambda - (((ndend-1)/2) * field_increment)
    total_cell_drive = np.zeros((res, res))
    
    # Pre-compute path arrays for vectorized spike generation
    p_x = path[:, 0]
    p_y = path[:, 1]
    dist_sq_path = (p_x - xxx)**2 + (p_y - yyy)**2
    sigma_env = target_lambda
    envelope_path = np.exp(-dist_sq_path / (2 * sigma_env**2))
    dist_sq_map = (X - xxx)**2 + (Y - yyy)**2
    envelope_map = np.exp(-dist_sq_map / (2 * sigma_env**2))
    
    for ni in range(ndend):
        ax = axes.flatten()[ni]
        ax2 = axes.flatten()[ni+ndend]

        offset_radius = 0.0
        theta_offset = ni * (2 * np.pi / ndend)
        x_offset = offset_radius * np.cos(theta_offset)
        y_offset = offset_radius * np.sin(theta_offset)

        l_var = lambda_base + ni * field_increment
        ang = grid_orientation + ni * (np.pi / 5.0)
        
        # Vectorized map generation
        Z_EC = vectorized_gridfield(ang, l_var, xxx + x_offset, yyy + y_offset, X, Y)
        Z_CA3 = Z_EC * envelope_map
        total_cell_drive += ((Z_EC + Z_CA3)/2) / ndend

        im = ax.imshow(Z_EC, origin='lower', extent=[0, myx, 0, myy], cmap='magma', vmin=0, vmax=1)
        im = ax2.imshow(Z_CA3, origin='lower', extent=[0, myx, 0, myy], cmap='magma', vmin=0, vmax=1)
        ax.set_title('Synapse %d' % (ni))
        ax2.set_title('Synapse %d' % (ni))

        ax.plot(path[:,0], path[:,1], color='white', alpha=0.4, linewidth=1)
        ax2.plot(path[:,0], path[:,1], color='white', alpha=0.4, linewidth=1)
        
        # Vectorized spike generation
        g_EC = vectorized_gridfield(ang, l_var, xxx + x_offset, yyy + y_offset, p_x, p_y)
        g_CA3 = g_EC * envelope_path
        
        prob_EC = g_EC * theta_mod
        prob_CA3 = g_CA3 * theta_mod
        
        # MnR glutamatergic inputs are not space, theta, or speed modulated, only novelty (global + location)
        prob_MnR = novelty_boost_arr # max is 1 x 1
        
        p_pos_EC = prob_EC * p_pos_EC_factor
        p_pos_CA3 = prob_CA3 * p_pos_CA3_factor
        p_neg_EC = prob_EC * p_neg_EC_factor
        p_neg_CA3 = prob_CA3 * p_neg_CA3_factor
        
        p_pos_EC[shock_record] = 1.0
        p_pos_CA3[shock_record] = 1.0
        p_neg_EC[shock_record] = 1.0
        p_neg_CA3[shock_record] = 1.0
        
        rnd_gate = np.random.rand(n_ticks)
        spike_prob_booster_MnR = 0.5
        spike_prob_booster_EC = 0.5
        spike_prob_booster_CA3 = 2.0
        
        cond1 = (g_EC > spatial_thresh_EC_arr) & (p_pos_EC * spike_prob_booster_EC > rnd_gate)
        cond2 = (g_CA3 > spatial_thresh_CA3_arr) & (p_pos_CA3 * spike_prob_booster_CA3 > rnd_gate)
        cond3 = (g_EC > spatial_thresh_EC_arr) & (p_neg_EC * spike_prob_booster_EC > rnd_gate)
        cond4 = (g_CA3 > spatial_thresh_CA3_arr) & (p_neg_CA3 * spike_prob_booster_CA3 > rnd_gate)
        
        # MnR uses no spatial threshold
        cond_MnR = (prob_MnR * spike_prob_booster_MnR > rnd_gate)
        
        spikes1 = np.where(cond1)[0]
        spikes2 = np.where(cond2)[0]
        spikes3 = np.where(cond3)[0]
        spikes4 = np.where(cond4)[0]
        spikes_MnR = np.where(cond_MnR)[0]
        
        global_idx = (my_field-1)*ndend + ni
        
        np.savetxt(dirname1 + '/g' + str(global_idx) + '_EC.txt', spikes1, fmt='%.0d')
        np.savetxt(dirname1 + '/g' + str(global_idx) + '_CA3.txt', spikes2, fmt='%.0d')
        np.savetxt(dirname2 + '/g' + str(global_idx) + '_EC.txt', spikes3, fmt='%.0d')
        np.savetxt(dirname2 + '/g' + str(global_idx) + '_CA3.txt', spikes4, fmt='%.0d')
        np.savetxt(dirname1 + '/g' + str(global_idx) + '_MnR.txt', spikes_MnR, fmt='%.0d')

    im_g = ax_g.imshow(total_cell_drive, origin='lower', extent=[0, myx, 0, myy], cmap='magma', vmin=0, vmax=1)
    ax_g.set_xlabel('x (cm)', fontsize=20)
    ax_g.set_ylabel('y (cm)', fontsize=20)
    cbar_g = fig_g.colorbar(im_g, ax=ax_g)
    cbar_g.set_label('Firing Probability', fontsize=18)
    fig_g.tight_layout()

    fig.text(0.5, 0.02, 'x (cm)', ha='center', fontsize=20)
    fig.text(0.02, 0.5, 'y (cm)', va='center', rotation='vertical', fontsize=20)
    plt.subplots_adjust(left=0.1, right=0.88, bottom=0.12, top=0.9, wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Firing Probability', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plot_dir = dirname1 + '/Probability_Maps/'
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    fig_g.savefig(plot_dir + 'Total_Field_%d_Prob.png' % my_field, dpi=150)
    fig.savefig(plot_dir + 'Field_%d_Prob.png' % my_field, dpi=150)
    fig.clf()
    fig_g.clf()
    plt.close(fig)
    plt.close(fig_g)
    
    print('Done with 2D Grid field mapped to PC group ' + str(my_field))

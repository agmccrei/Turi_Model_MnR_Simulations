import csv
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import scipy.stats as stats
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# Matplotlib configuration for publication
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,           # 8pt base size for all general text
    'axes.labelsize': 7,      # 8pt for X and Y axis labels
    'xtick.labelsize': 7,     # 8pt for X axis tick numbers
    'ytick.labelsize': 7,     # 8pt for Y axis tick numbers
    'axes.titlesize': 14,     # 14pt for Main Graphic/Caption Title Header
    'pdf.fonttype': 42,       # Keeps Arial fully editable in Illustrator/LaTeX
    'ps.fonttype': 42
})

label_fs = 7
title_fs = 7
tick_fs = 7
panel_label_fs = 14
example2plot = 1 # 1 & 6 seem like good examples

def cohen_d(y, x):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def compare_rates(rates1, rates2):
    rates1 = np.array(rates1)
    rates2 = np.array(rates2)
    if len(rates1) < 2 or len(rates2) < 2:
        return np.nan, np.nan, np.nan

    try:
        t_stat, p_val = stats.ttest_rel(rates1, rates2)
        cd = cohen_d(rates1, rates2)
    except Exception:
        t_stat = np.nan
        p_val = np.nan
        cd = np.nan

    return t_stat, p_val, cd


def summarize_group(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean_val = np.mean(values)
    sd_val = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return mean_val, sd_val


def run_shapiro_wilk(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    if np.allclose(values, values[0]):
        return np.nan, np.nan
    try:
        return stats.shapiro(values)
    except Exception:
        return np.nan, np.nan


def run_wilcoxon_signed_rank(rates1, rates2):
    rates1 = np.asarray(rates1, dtype=float)
    rates2 = np.asarray(rates2, dtype=float)
    if len(rates1) < 2 or len(rates2) < 2 or len(rates1) != len(rates2):
        return np.nan, np.nan
    try:
        return stats.wilcoxon(rates1, rates2, zero_method='wilcox', correction=False)
    except Exception:
        return np.nan, np.nan


def analyze_paired_comparison(rates1, rates2):
    rates1 = np.asarray(rates1, dtype=float)
    rates2 = np.asarray(rates2, dtype=float)
    t_stat, p_val, cd = compare_rates(rates1, rates2)
    diff_vals = rates2 - rates1
    shapiro_stat_diff, shapiro_p_diff = run_shapiro_wilk(diff_vals)
    wilcoxon_stat, wilcoxon_p_val = run_wilcoxon_signed_rank(rates1, rates2)
    if (not np.isnan(shapiro_p_diff)) and shapiro_p_diff < 0.05:
        effective_p_val = wilcoxon_p_val
    else:
        effective_p_val = p_val
    return t_stat, p_val, cd, wilcoxon_stat, wilcoxon_p_val, shapiro_stat_diff, shapiro_p_diff, effective_p_val


def _format_csv_value(value):
    if isinstance(value, (np.floating, float)):
        return '' if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_stats_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        with open(output_path, 'w', newline='') as f:
            f.write('')
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(value) for key, value in row.items()})


def load_data_mnr(group_name, trial_idx):
    """Load data from a specific group and trial for the MnR experiment."""
    res_dir = f"Simulation_Results/2D_box_Familiar2Novelty/{group_name}/Trial_{trial_idx}/Run_{trial_idx}"
    spike_file = f"{res_dir}/spikes.dat"
    path_file = f"{res_dir}/path.txt"
    velocity_file = f"make_inputs_linear_track/runs_produced_2D_cond4_pos/run_{trial_idx}/velocity.txt"

    if os.path.exists(spike_file) and os.path.exists(path_file):
        try:
            spikes = np.loadtxt(spike_file)
            if spikes.ndim == 1:
                if len(spikes) > 0:
                    spikes = np.array([spikes])
                else:
                    spikes = np.empty((0, 2))
            path = np.loadtxt(path_file)
            velocity = np.loadtxt(velocity_file) if os.path.exists(velocity_file) else None
            return spikes, path, velocity
        except Exception as e:
            print(f"Error loading data from {res_dir}: {e}")
            return np.empty((0, 2)), np.empty((0, 2)), None
    return np.empty((0, 2)), np.empty((0, 2)), None


# Network constants
DT = 0.1
TINIT = 400
STARTDEL = 500
THETA = 125
duration = 176
SIMDUR = TINIT + (THETA * duration) + STARTDEL / 10
number_of_trials = 15

nplf = 49
ndend = 8
scale = 1
npcell = 130 * scale
aacell = 2 * scale
nbcell = 8 * scale
nbscell = 2 * scale
nolm = 2 * scale
nvipcck = 1 * scale
nvipcr = 4 * scale
nvipcrnvm = 1 * scale

iPC = 0
iAAC = npcell
iBC = npcell + aacell
iBSC = npcell + aacell + nbcell
iOLM = npcell + aacell + nbcell + nbscell
iVCCK = npcell + aacell + nbcell + nbscell + nolm
iVCR = npcell + aacell + nbcell + nbscell + nolm + nvipcck
iVCRnvm = npcell + aacell + nbcell + nbscell + nolm + nvipcck + nvipcr
iCA3 = iVCRnvm + nvipcrnvm

# Cell-group definitions
cell_groups = [
    ('PC', np.arange(iPC, iAAC), 'dimgray'),
    ('AAC', np.arange(iAAC, iBC), 'red'),
    ('PV-BC', np.arange(iBC, iBSC), 'orange'),
    ('BSC', np.arange(iBSC, iOLM), 'brown'),
    ('OLM', np.arange(iOLM, iVCCK), 'magenta'),
    ('VIP/CCK-BC', np.arange(iVCCK, iVCR), 'cyan'),
    ('VIP/CR-IS-3', np.arange(iVCR, iVCRnvm), 'blue'),
    ('VIP-NVM', np.arange(iVCRnvm, iCA3), 'purple')
]

conditions_mnr = [
    ("Control", "Control"),
    ("MnR_removed", "Silenced_MnR"),
    ("MnR_removed_VCpvm", "Silenced_MnR_to_VCpvm"),
    ("MnR_removed_CCK_VCnvm", "Silenced_MnR_to_CCK_VCnvm")
]

bin_width_ms = 500.0
bin_width_s = bin_width_ms / 1000.0
bins_ms = np.arange(STARTDEL, SIMDUR + bin_width_ms, bin_width_ms)
bins_s = bins_ms / 1000.0

min_spikes = 5
grid_res = 20
num_fields = 130

def build_rate_map(valid_path, spike_times, grid_res=grid_res):
    if len(valid_path) < 2:
        return np.zeros((grid_res, grid_res), dtype=float), np.zeros((grid_res, grid_res), dtype=float)

    x_coords = valid_path[:, 0]
    y_coords = valid_path[:, 1]
    occupancy, _, _ = np.histogram2d(x_coords, y_coords, bins=grid_res, range=[[0, 100], [0, 100]])
    occupancy = occupancy.astype(float)

    if len(spike_times) == 0:
        return occupancy, np.zeros_like(occupancy)

    spike_coords = valid_path[spike_times]
    if len(spike_coords) == 0:
        return occupancy, np.zeros_like(occupancy)

    spike_counts, _, _ = np.histogram2d(spike_coords[:, 0], spike_coords[:, 1], bins=grid_res, range=[[0, 100], [0, 100]])
    return occupancy, spike_counts.astype(float)


def compute_spatial_coherence(occupancy, spike_counts):
    visited = occupancy > 0
    if np.sum(visited) < 4:
        return np.nan

    rate_map = np.zeros_like(occupancy, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map[visited] = spike_counts[visited] / occupancy[visited]
    rate_map = np.nan_to_num(rate_map, nan=0.0)

    center_vals = []
    neighbor_vals = []
    for r in range(1, rate_map.shape[0] - 1):
        for c in range(1, rate_map.shape[1] - 1):
            if not visited[r, c]:
                continue
            neighbors = []
            for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if visited[rr, cc]:
                    neighbors.append(rate_map[rr, cc])
            if len(neighbors) >= 1:
                center_vals.append(rate_map[r, c])
                neighbor_vals.append(np.mean(neighbors))

    if len(center_vals) < 3:
        return np.nan

    try:
        corr = np.corrcoef(center_vals, neighbor_vals)[0, 1]
    except Exception:
        corr = np.nan

    return corr if np.isfinite(corr) else np.nan


def classify_place_cell(spikes, path, t_start, t_end, n_shuffles=100):
    if spikes is None or len(spikes) == 0 or len(path) == 0:
        return False, np.nan, False, np.nan

    idx_start = int(max(0, t_start))
    idx_end = int(min(len(path), t_end))
    valid_path = path[idx_start:idx_end] if idx_start < idx_end else np.empty((0, 2))
    if len(valid_path) < 10:
        return False, np.nan, False, np.nan

    valid_spikes = spikes[(spikes[:, 0] >= idx_start) & (spikes[:, 0] < idx_end)]
    if len(valid_spikes) < min_spikes:
        return False, np.nan, False, np.nan

    spike_times = valid_spikes[:, 0].astype(int)
    spike_time_indices = (spike_times - idx_start).astype(int)
    spike_time_indices = spike_time_indices[(spike_time_indices >= 0) & (spike_time_indices < len(valid_path))]
    if len(spike_time_indices) < min_spikes:
        return False, np.nan, False, np.nan

    spike_coords = valid_path[spike_time_indices]
    if len(spike_coords) < min_spikes:
        return False, np.nan, False, np.nan

    # DBSCAN on the raw spike coordinates to identify dense clusters.
    spatial_spread = 20.0
    max_spread = 40
    db = DBSCAN(eps=spatial_spread, min_samples=min_spikes)
    labels = db.fit_predict(spike_coords)
    cluster_mask = labels != -1
    cluster_found = False
    if np.any(cluster_mask):
        cluster_sizes = []
        for cluster_id in np.unique(labels[cluster_mask]):
            cluster_points = spike_coords[labels == cluster_id]
            if len(cluster_points) >= min_spikes:
                bbox_w = np.ptp(cluster_points[:, 0])
                bbox_h = np.ptp(cluster_points[:, 1])
                if (bbox_w <= max_spread) and (bbox_h <= max_spread) and ((bbox_w * bbox_h) <= max_spread**2):
                    cluster_found = True
                    break

    occupancy, spike_counts = build_rate_map(valid_path, spike_time_indices, grid_res=grid_res)
    observed_coherence = compute_spatial_coherence(occupancy, spike_counts)

    rng = np.random.default_rng(7)
    shuffled_scores = []
    segment_length = len(valid_path)
    for _ in range(n_shuffles):
        shift_ms = int(rng.uniform(2000, 5000))
        shifted_indices = np.mod(spike_time_indices - shift_ms, segment_length)
        shifted_coords = valid_path[shifted_indices]
        if len(shifted_coords) < min_spikes:
            continue
        shuffle_occ, shuffle_counts = build_rate_map(valid_path, shifted_indices, grid_res=grid_res)
        shuffle_score = compute_spatial_coherence(shuffle_occ, shuffle_counts)
        if np.isfinite(shuffle_score):
            shuffled_scores.append(shuffle_score)

    if len(shuffled_scores) == 0 or not np.isfinite(observed_coherence):
        return False, np.nan, cluster_found, np.nan

    shuffled_scores = np.array(shuffled_scores)
    p_val = (np.sum(shuffled_scores >= observed_coherence) + 1) / (len(shuffled_scores) + 1)
    coherence_ok = (p_val < 0.05) and np.isfinite(observed_coherence)
    place_cell = coherence_ok and (cluster_found or observed_coherence > 0.15)
    return place_cell, p_val, cluster_found, observed_coherence


def get_place_cell_ids_for_intervals(spikes, path, intervals):
    if spikes is None or len(spikes) == 0 or len(path) == 0:
        return {interval: [] for interval in intervals}

    interval_results = {interval: [] for interval in intervals}
    for cell_id in range(iPC, iPC + num_fields):
        cell_mask = spikes[:, 1] == cell_id
        cell_spikes = spikes[cell_mask, 0].astype(int)
        if len(cell_spikes) < min_spikes:
            continue

        for interval in intervals:
            t_start, t_end = interval
            filtered_spikes = cell_spikes[(cell_spikes >= t_start) & (cell_spikes < t_end)]
            if len(filtered_spikes) < min_spikes:
                continue

            spike_series = np.column_stack([filtered_spikes, np.full(len(filtered_spikes), cell_id)])
            is_place_cell, _, _, _ = classify_place_cell(spike_series, path, t_start, t_end)
            if is_place_cell:
                interval_results[interval].append(cell_id)

    return interval_results


def compute_place_cell_metrics(spikes, path):
    def get_metrics_for_interval(t_start, t_end):
        total_dist_m = 0.0
        downsample = 50
        if len(path) > 0:
            segment = path[int(max(0, t_start)):int(min(len(path), t_end)):downsample]
            if len(segment) > 1:
                diffs = np.diff(segment, axis=0)
                total_dist_m += np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))) / 100.0

        if total_dist_m == 0:
            total_dist_m = 1.0

        return {
            'num_pcs': len(place_cell_ids_by_interval[(t_start, t_end)]),
            'num_pcs_density': len(place_cell_ids_by_interval[(t_start, t_end)]) / total_dist_m,
            'place_cell_ids': place_cell_ids_by_interval[(t_start, t_end)]
        }

    intervals = [(STARTDEL, 11500), (11500, SIMDUR), (STARTDEL, SIMDUR)]
    place_cell_ids_by_interval = get_place_cell_ids_for_intervals(spikes, path, intervals)

    metrics_fam = get_metrics_for_interval(STARTDEL, 11500)
    metrics_nov = get_metrics_for_interval(11500, SIMDUR)
    metrics_full = get_metrics_for_interval(STARTDEL, SIMDUR)

    return {
        'num_pcs': metrics_nov['num_pcs'] - metrics_fam['num_pcs'],
        'num_pcs_density': metrics_nov['num_pcs_density'] - metrics_fam['num_pcs_density'],
        'num_pcs_familiar': metrics_fam['num_pcs'],
        'num_pcs_density_familiar': metrics_fam['num_pcs_density'],
        'num_pcs_novel': metrics_nov['num_pcs'],
        'num_pcs_density_novel': metrics_nov['num_pcs_density'],
        'place_cell_ids': metrics_nov['place_cell_ids'],
        'full_place_cell_ids': metrics_full['place_cell_ids']
    }


print("Loading MnR removal data...")

all_data_mnr = {}
all_metrics_mnr = {}
max_rates = {name: 0.0 for name, _, _ in cell_groups}

for cond_label, cond_dir in conditions_mnr:
    all_data_mnr[cond_label] = {}
    all_metrics_mnr[cond_label] = {}

    for trial_idx in range(number_of_trials):
        spikes, path, velocity = load_data_mnr(cond_dir, trial_idx)
        all_data_mnr[cond_label][trial_idx] = (spikes, path, velocity)
        metrics = compute_place_cell_metrics(spikes, path)
        all_metrics_mnr[cond_label][trial_idx] = metrics

        for name, gids, _ in cell_groups:
            if spikes is not None and len(spikes) > 0:
                group_spikes = spikes[np.isin(spikes[:, 1], gids)]
                if len(group_spikes) > 0 and len(gids) > 0:
                    counts, _ = np.histogram(group_spikes[:, 0], bins=bins_ms)
                    avg_rate = counts / (bin_width_s * len(gids))
                    current_max = np.max(avg_rate)
                    if current_max > max_rates[name]:
                        max_rates[name] = current_max

for k in max_rates:
    if max_rates[k] < 0.1:
        max_rates[k] = 1.0
    else:
        max_rates[k] *= 1.1

print("Computing spike rates for all runs...")
spike_rates_all_runs = {}
spike_rates_fam_all_runs = {}
spike_rates_nov_all_runs = {}

for cond_label, cond_dir in conditions_mnr:
    spike_rates_all_runs[cond_label] = {}
    spike_rates_fam_all_runs[cond_label] = {}
    spike_rates_nov_all_runs[cond_label] = {}
    for trial_idx in range(number_of_trials):
        spikes, path, velocity = all_data_mnr[cond_label][trial_idx]
        spike_rates_all_runs[cond_label][trial_idx] = {}
        spike_rates_fam_all_runs[cond_label][trial_idx] = {}
        spike_rates_nov_all_runs[cond_label][trial_idx] = {}

        time_fam_s = (11500 - STARTDEL) / 1000.0
        time_nov_s = (SIMDUR - 11500) / 1000.0

        for name, gids, _ in cell_groups:
            fam_rates = []
            nov_rates = []
            delta_rates = []
            if spikes is not None and len(spikes) > 0:
                for gid in gids:
                    cell_spikes = spikes[spikes[:, 1] == gid, 0]
                    count_fam = np.sum((cell_spikes >= STARTDEL) & (cell_spikes < 11500))
                    rate_fam = count_fam / time_fam_s
                    count_nov = np.sum((cell_spikes >= 11500) & (cell_spikes < SIMDUR))
                    rate_nov = count_nov / time_nov_s
                    fam_rates.append(rate_fam)
                    nov_rates.append(rate_nov)
                    delta_rates.append(rate_nov - rate_fam)
            else:
                fam_rates = [0.0] * len(gids)
                nov_rates = [0.0] * len(gids)
                delta_rates = [0.0] * len(gids)

            spike_rates_all_runs[cond_label][trial_idx][name] = np.array(delta_rates)
            spike_rates_fam_all_runs[cond_label][trial_idx][name] = np.array(fam_rates)
            spike_rates_nov_all_runs[cond_label][trial_idx][name] = np.array(nov_rates)

stats_rows = []
familiar_novelty_rows = []

for group_name, _, _ in cell_groups:
    for i, (cond_label, _) in enumerate(conditions_mnr):
        if i == 0:
            continue
        ref_vals = [np.mean(spike_rates_all_runs[conditions_mnr[0][0]][trial_idx][group_name]) for trial_idx in range(number_of_trials)]
        cmp_vals = [np.mean(spike_rates_all_runs[cond_label][trial_idx][group_name]) for trial_idx in range(number_of_trials)]
        mean_a, sd_a = summarize_group(ref_vals)
        mean_b, sd_b = summarize_group(cmp_vals)
        t_stat, p_val, cd, wilcoxon_stat, wilcoxon_p_val, shapiro_stat_diff, shapiro_p_diff, _ = analyze_paired_comparison(ref_vals, cmp_vals)
        shapiro_stat_a, shapiro_p_a = run_shapiro_wilk(ref_vals)
        shapiro_stat_b, shapiro_p_b = run_shapiro_wilk(cmp_vals)
        stats_rows.append({
            'metric': 'rate_delta',
            'group': group_name,
            'comparison_type': 'condition_vs_condition',
            'condition_a': conditions_mnr[0][0],
            'condition_b': cond_label,
            'mean_a': mean_a,
            'sd_a': sd_a,
            'mean_b': mean_b,
            'sd_b': sd_b,
            'statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cd,
            'shapiro_stat_a': shapiro_stat_a,
            'shapiro_p_a': shapiro_p_a,
            'shapiro_stat_b': shapiro_stat_b,
            'shapiro_p_b': shapiro_p_b,
            'shapiro_stat_diff': shapiro_stat_diff,
            'shapiro_p_diff': shapiro_p_diff,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p_val,
            'n_trials': len(ref_vals)
        })

    for cond_label, _ in conditions_mnr:
        fam_vals = [np.mean(spike_rates_fam_all_runs[cond_label][trial_idx][group_name]) for trial_idx in range(number_of_trials)]
        nov_vals = [np.mean(spike_rates_nov_all_runs[cond_label][trial_idx][group_name]) for trial_idx in range(number_of_trials)]
        mean_period_a, sd_period_a = summarize_group(fam_vals)
        mean_period_b, sd_period_b = summarize_group(nov_vals)
        t_stat, p_val, cd, wilcoxon_stat, wilcoxon_p_val, shapiro_stat_diff, shapiro_p_diff, _ = analyze_paired_comparison(fam_vals, nov_vals)
        shapiro_stat_a, shapiro_p_a = run_shapiro_wilk(fam_vals)
        shapiro_stat_b, shapiro_p_b = run_shapiro_wilk(nov_vals)
        familiar_novelty_rows.append({
            'metric': 'rate',
            'group': group_name,
            'condition': cond_label,
            'comparison_type': 'familiar_vs_novelty',
            'period_a': 'familiar',
            'period_b': 'novelty',
            'mean_period_a': mean_period_a,
            'sd_period_a': sd_period_a,
            'mean_period_b': mean_period_b,
            'sd_period_b': sd_period_b,
            'statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cd,
            'shapiro_stat_period_a': shapiro_stat_a,
            'shapiro_p_period_a': shapiro_p_a,
            'shapiro_stat_period_b': shapiro_stat_b,
            'shapiro_p_period_b': shapiro_p_b,
            'shapiro_stat_diff': shapiro_stat_diff,
            'shapiro_p_diff': shapiro_p_diff,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p_val,
            'n_trials': len(fam_vals)
        })

for cond_label, _ in conditions_mnr:
    density_fam = [all_metrics_mnr[cond_label][trial_idx]['num_pcs_density_familiar'] for trial_idx in range(number_of_trials)]
    density_nov = [all_metrics_mnr[cond_label][trial_idx]['num_pcs_density_novel'] for trial_idx in range(number_of_trials)]
    mean_period_a, sd_period_a = summarize_group(density_fam)
    mean_period_b, sd_period_b = summarize_group(density_nov)
    t_stat, p_val, cd, wilcoxon_stat, wilcoxon_p_val, shapiro_stat_diff, shapiro_p_diff, _ = analyze_paired_comparison(density_fam, density_nov)
    shapiro_stat_a, shapiro_p_a = run_shapiro_wilk(density_fam)
    shapiro_stat_b, shapiro_p_b = run_shapiro_wilk(density_nov)
    familiar_novelty_rows.append({
        'metric': 'place_cell_density',
        'group': 'pyramidal_cells',
        'condition': cond_label,
        'comparison_type': 'familiar_vs_novelty',
        'period_a': 'familiar',
        'period_b': 'novelty',
        'mean_period_a': mean_period_a,
        'sd_period_a': sd_period_a,
        'mean_period_b': mean_period_b,
        'sd_period_b': sd_period_b,
        'statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cd,
        'shapiro_stat_period_a': shapiro_stat_a,
        'shapiro_p_period_a': shapiro_p_a,
        'shapiro_stat_period_b': shapiro_stat_b,
        'shapiro_p_period_b': shapiro_p_b,
        'shapiro_stat_diff': shapiro_stat_diff,
        'shapiro_p_diff': shapiro_p_diff,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_p_value': wilcoxon_p_val,
        'n_trials': len(density_fam)
    })

for i, (cond_label, _) in enumerate(conditions_mnr):
    if i == 0:
        continue
    ref_vals = [all_metrics_mnr[conditions_mnr[0][0]][trial_idx]['num_pcs_density'] for trial_idx in range(number_of_trials)]
    cmp_vals = [all_metrics_mnr[cond_label][trial_idx]['num_pcs_density'] for trial_idx in range(number_of_trials)]
    mean_a, sd_a = summarize_group(ref_vals)
    mean_b, sd_b = summarize_group(cmp_vals)
    t_stat, p_val, cd, wilcoxon_stat, wilcoxon_p_val, shapiro_stat_diff, shapiro_p_diff, _ = analyze_paired_comparison(ref_vals, cmp_vals)
    shapiro_stat_a, shapiro_p_a = run_shapiro_wilk(ref_vals)
    shapiro_stat_b, shapiro_p_b = run_shapiro_wilk(cmp_vals)
    stats_rows.append({
        'metric': 'place_cell_density',
        'group': 'pyramidal_cells',
        'comparison_type': 'condition_vs_condition',
        'condition_a': conditions_mnr[0][0],
        'condition_b': cond_label,
        'mean_a': mean_a,
        'sd_a': sd_a,
        'mean_b': mean_b,
        'sd_b': sd_b,
        'statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cd,
        'shapiro_stat_a': shapiro_stat_a,
        'shapiro_p_a': shapiro_p_a,
        'shapiro_stat_b': shapiro_stat_b,
        'shapiro_p_b': shapiro_p_b,
        'shapiro_stat_diff': shapiro_stat_diff,
        'shapiro_p_diff': shapiro_p_diff,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_p_value': wilcoxon_p_val,
        'n_trials': len(ref_vals)
    })

os.makedirs('Plots', exist_ok=True)
save_stats_csv(stats_rows, 'Plots/plot_MnR_removal_place_cells_stats.csv')
save_stats_csv(familiar_novelty_rows, 'Plots/plot_MnR_removal_place_cells_familiar_novelty_stats.csv')


def add_panel_label(ax, label):
    ax.annotate(
        label,
        xy=(0, 1),
        xytext=(-30, 45) if label in ['a','c'] else (-30, 13),
        xycoords='axes fraction',
        textcoords='offset points',
        fontsize=panel_label_fs,
        fontweight='bold',
        ha='left',
        va='top',
        annotation_clip=False,
    )


import matplotlib.patches as mpatches

print("Drawing figure...")
target_width = 7.0 
target_height = 5.0
fig = plt.figure(figsize=(target_width, target_height), layout='constrained', dpi=300)
fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02)

# Main layout: Left column (A/B), Right column (C & D)
gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 3.25])

# Left column: A (schematic) and B (PSTHs & speed)
gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.75, 2.25])

ax_A = fig.add_subplot(gs_left[0, 0])
ax_A.axis('off')
ax_A.set_aspect('auto', adjustable='box', anchor='W')

add_panel_label(ax_A, 'a')
try:
    img = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Circuit_Schematic_V5.png'))
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0.05
    else:
        mask = np.any(img[:, :, :3] < 0.95, axis=2)
    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img[rmin:rmax + 1, cmin:cmax + 1, :]
    ax_A.imshow(img, extent=[-0.23, 1.3, -0.4, 1.5], aspect='auto', clip_on=False)
    ax_A.set_xlim(0, 1)
    ax_A.set_ylim(0, 1)
except Exception as e:
    print(f"Warning: Could not load Circuit_Schematic_V5.png - {e}")
    ax_A.text(0.5, 0.5, 'Circuit Schematic V5\n(image not found)', ha='center', va='center', fontsize=18)

# Panel B: PSTH and speed plots
psth_groups = [
    ('PC', 'PC', np.arange(iPC, iAAC), 'dimgray'),
    ('PV-BC', 'PV-BC', np.arange(iBC, iBSC), 'orange'),
    ('OLM', 'OLM', np.arange(iOLM, iVCCK), 'magenta'),
    ('VIP/CR-IS-3', 'VIP/CR-IS-3', np.arange(iVCR, iVCRnvm), 'blue'),
    ('VIP/CCK-BC', 'VIP/CCK-BC', np.arange(iVCCK, iVCR), 'cyan'),
    ('VIP-NVM', 'VIP-NVM', np.arange(iVCRnvm, iCA3), 'purple')
]

gs_B = gridspec.GridSpecFromSubplotSpec(len(psth_groups) + 1, 1, subplot_spec=gs_left[1, 0], hspace=0.05)

for row_idx, (cell_label, group_key, gids, color) in enumerate(psth_groups):
    ax = fig.add_subplot(gs_B[row_idx, 0])
    if row_idx == 0:
        add_panel_label(ax, 'b')
        ax.set_title('Familiar                Novel', fontsize=title_fs, pad=0)
    spikes, _, velocity = all_data_mnr['Control'][example2plot]
    if spikes is not None and len(spikes) > 0:
        group_spikes = spikes[np.isin(spikes[:, 1], gids)]
        if len(group_spikes) > 0 and len(gids) > 0:
            counts, _ = np.histogram(group_spikes[:, 0], bins=bins_ms)
            avg_rate = counts / (bin_width_s * len(gids))
            ax.bar(bins_s[:-1], avg_rate, width=bin_width_s, align='edge', color=color, alpha=0.7)
    ax.set_xlim(STARTDEL / 1000., SIMDUR / 1000.)
    ax.axvline(x=11.5, color='k', linestyle='--', linewidth=1, zorder=2)
    ax.set_ylim(0, max_rates[group_key])
    ax.set_ylabel('[Hz]', fontsize=label_fs, labelpad=0)
    ax.text(0.03, 0.85, cell_label, transform=ax.transAxes, ha='left', va='top', fontsize=tick_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs, pad=0)
    ax.set_xticks([])

ax_speed = fig.add_subplot(gs_B[len(psth_groups), 0])
ax_speed.spines['right'].set_visible(False)
ax_speed.spines['top'].set_visible(False)
if velocity is not None:
    time_axis_ms = np.arange(len(velocity)) / 1000.
    ax_speed.plot(time_axis_ms, velocity * 1000, color='green', linewidth=1)
ax_speed.set_xlim(STARTDEL / 1000., SIMDUR / 1000.)
ax_speed.axvline(x=11.5, color='k', linestyle='--', linewidth=1, zorder=2)
ax_speed.set_ylim(bottom=0)
ax_speed.set_ylabel('[cm/s]', fontsize=label_fs, labelpad=0)
ax_speed.set_xlabel('Time (s)', fontsize=label_fs, labelpad=0)
ax_speed.tick_params(axis='both', which='major', labelsize=tick_fs, pad=0)

# Panel C: Delta rate box plots
gs_C = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_main[0, 1])
delta_groups = [
    ('VIP/CR-IS-3', 'VIP/CR-IS-3', np.arange(iVCR, iVCRnvm), 'blue'),
    ('VIP/CCK-BC', 'VIP/CCK-BC', np.arange(iVCCK, iVCR), 'cyan'),
    ('VIP-NVM', 'VIP-NVM', np.arange(iVCRnvm, iCA3), 'purple'),
    ('OLM', 'OLM', np.arange(iOLM, iVCCK), 'magenta'),
    ('PV-BC', 'PV-BC', np.arange(iBC, iBSC), 'orange'),
    ('PC', 'PC', np.arange(iPC, iAAC), 'dimgray')
]

x_pos = np.arange(len(conditions_mnr))
alphas = [1.0, 1.0, 1.0, 1.0]
box_hatches = ['', '', '////', 'xxxx']

for panel_idx, (display_name, group_key, gids, color) in enumerate(delta_groups):
    ax = fig.add_subplot(gs_C[panel_idx // 3, panel_idx % 3])
    if panel_idx == 0:
        add_panel_label(ax, 'c')

    all_vals = []
    for cond_label, _ in conditions_mnr:
        run_vals = []
        for trial_idx in range(number_of_trials):
            rates = spike_rates_all_runs[cond_label][trial_idx][group_key] if group_key in spike_rates_all_runs[cond_label][trial_idx] else np.array([])
            if len(rates) > 0:
                run_vals.append(np.mean(rates))
        all_vals.append(np.array(run_vals))

    for i, (cond_label, _) in enumerate(conditions_mnr):
        dots = all_vals[i]
        if len(dots) == 0:
            continue
        facecolor = 'white' if i != 0 else to_rgba(color, alpha=alphas[i])
        bp = ax.boxplot(dots, positions=[i], widths=0.7, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=facecolor, edgecolor='black', linewidth=1, hatch=box_hatches[i]),
                        medianprops=dict(color='black', linewidth=1),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))
        x_jitter_cond = np.random.normal(i, 0.0, size=len(dots))
        ax.scatter(x_jitter_cond, dots, s=8, alpha=0.6, color='k', edgecolor='none', zorder=3)

        if i > 0:
            prev_dots = all_vals[i - 1]
            if len(prev_dots) > 0:
                x_jitter_prev = np.random.normal(i - 1, 0.0, size=len(prev_dots))
                for xc, xm, yc, ym in zip(x_jitter_prev, x_jitter_cond, prev_dots, dots):
                    ax.plot([xc, xm], [yc, ym], color='dimgrey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    y_max_data = max([np.max(vals) for vals in all_vals if len(vals) > 0]) if any(len(vals) > 0 for vals in all_vals) else 1.0
    y_min_data = min([np.min(vals) for vals in all_vals if len(vals) > 0]) if any(len(vals) > 0 for vals in all_vals) else 0.0
    y_range = max(0.1, y_max_data - y_min_data)
    y_star = max(0, y_max_data) + y_range * 0.05
    y_step = y_range * 0.35

    for i, (cond_label, _) in enumerate(conditions_mnr[1:], 1):
        control_vals = all_vals[0]
        cond_vals = all_vals[i]
        _, p_val, cd, _, wilcoxon_p_val, _, shapiro_p_diff, effective_p_val = analyze_paired_comparison(control_vals, cond_vals)
        sig_p_val = wilcoxon_p_val if (not np.isnan(shapiro_p_diff)) and shapiro_p_diff < 0.05 else p_val
        if not np.isnan(sig_p_val) and sig_p_val < 0.05:
            ax.plot([0, i], [y_star, y_star], color='black', linewidth=1)
            arrow = '↑' if cd > 0 else '↓'
            ax.text(i / 2.0, y_star, '*', ha='center', va='bottom', fontsize=12, fontweight='bold')
            y_star += y_step

    if panel_idx in (0, 3):
        ax.set_ylabel(r'$\Delta$'+'Novel-Familiar (Hz)', fontsize=label_fs, labelpad=0)
    ax.set_ylim(bottom=min(0, y_min_data) - y_range * 0.1, top=y_star + y_range * 0.24 - y_step)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_title(display_name, fontsize=title_fs, pad=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=tick_fs, pad=0)
    ax.tick_params(axis='x', which='major', length=0)

# Panel D: Place cell density box plot
ax_D = fig.add_subplot(gs_C[2, :])
add_panel_label(ax_D, 'd')

pc_metrics = {cond: {'num_pcs_density': []} for cond, _ in conditions_mnr}
for cond_label, _ in conditions_mnr:
    for trial_idx in range(number_of_trials):
        metrics = all_metrics_mnr[cond_label][trial_idx]
        pc_metrics[cond_label]['num_pcs_density'].append(metrics['num_pcs_density'])

num_data = [pc_metrics[cond]['num_pcs_density'] for cond, _ in conditions_mnr]
np.random.seed(100)
for i, vals in enumerate(num_data):
    facecolor = 'white' if i != 0 else to_rgba('dimgray', alpha=alphas[i])
    bp = ax_D.boxplot(vals, positions=[i], widths=0.7, patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=facecolor, edgecolor='black', linewidth=1, hatch=box_hatches[i]),
                      medianprops=dict(color='black', linewidth=1),
                      whiskerprops=dict(linewidth=1),
                      capprops=dict(linewidth=1))
    x_jitter_cond = np.random.normal(i, 0.0, size=len(vals))
    ax_D.scatter(x_jitter_cond, vals, s=8, alpha=0.6, color='k', edgecolor='none', zorder=3)

    if i > 0:
        prev_vals = num_data[i - 1]
        x_jitter_prev = np.random.normal(i - 1, 0.0, size=len(prev_vals))
        for xc, xm, yc, ym in zip(x_jitter_prev, x_jitter_cond, prev_vals, vals):
            ax_D.plot([xc, xm], [yc, ym], color='dimgrey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

all_vals = [val for vals in num_data for val in vals]
if len(all_vals) > 0:
    y_max_data = max(all_vals)
    y_min_data = min(all_vals)
else:
    y_max_data, y_min_data = 1.0, 0.0

y_range = max(0.1, y_max_data - y_min_data)
y_star = max(0, y_max_data) + y_range * 0.05
y_step = y_range * 0.25
for i, (cond_label, _) in enumerate(conditions_mnr[1:], 1):
    _, p_val, cd, _, wilcoxon_p_val, _, shapiro_p_diff, effective_p_val = analyze_paired_comparison(num_data[0], num_data[i])
    sig_p_val = wilcoxon_p_val if (not np.isnan(shapiro_p_diff)) and shapiro_p_diff < 0.05 else p_val
    if not np.isnan(sig_p_val) and sig_p_val < 0.05:
        ax_D.plot([0, i], [y_star, y_star], color='black', linewidth=1)
        arrow = '↑' if cd > 0 else '↓'
        ax_D.text(i / 2.0, y_star, '*', ha='center', va='bottom', fontsize=12, fontweight='bold')
        y_star += y_step

ax_D.set_ylim(bottom=min(0, y_min_data) - y_range * 0.1, top=y_star + y_range * 0.2 - y_step)
ax_D.axhline(0, xmin=0.0, xmax=0.4, color='gray', linestyle='--', linewidth=1, zorder=1)
ax_D.set_xticks(x_pos)
ax_D.set_xticklabels([])
ax_D.set_xlim(right=10)
ax_D.set_ylabel(r'$\Delta$'+'Novel-Familiar\n(place cells/m)', fontsize=label_fs, labelpad=0)
ax_D.spines['right'].set_visible(False)
ax_D.spines['top'].set_visible(False)
ax_D.tick_params(axis='y', which='major', labelsize=tick_fs, pad=0)
ax_D.tick_params(axis='x', which='major', length=0)

example2plot_2 = 1 # consider 14
ax_inset = ax_D.inset_axes([0.15, 0.1, 0.825, 0.825]) 
ax_inset2 = ax_D.inset_axes([0.45, 0.1, 0.825, 0.825]) 
spikes, path, velocity = all_data_mnr['Control'][example2plot_2]
spikes2, path2, velocity2 = all_data_mnr['MnR_removed'][example2plot_2]
if len(path) > 0:
    ax_inset.plot(path[STARTDEL:int(SIMDUR), 0], path[STARTDEL:int(SIMDUR), 1], 'dimgray', alpha=1.0, linewidth=1.0, zorder=0)
    ax_inset2.plot(path2[STARTDEL:int(SIMDUR), 0], path2[STARTDEL:int(SIMDUR), 1], 'dimgray', alpha=1.0, linewidth=1.0, zorder=0)

    place_cell_ids = all_metrics_mnr['Control'][example2plot_2]['full_place_cell_ids']
    valid_pcs = len(place_cell_ids)
    top_cell_ids = place_cell_ids[:20]
    if len(top_cell_ids) > 0:
        colors = plt.cm.turbo(np.linspace(0., 1., len(top_cell_ids)))
        for i, cell_id in enumerate(top_cell_ids):
            cell_mask = spikes[:, 1] == cell_id
            pc_spikes = spikes[cell_mask, 0].astype(int)
            spike_times = pc_spikes[(pc_spikes >= STARTDEL) & (pc_spikes < len(path))]
            if len(spike_times) > 0:
               ax_inset.scatter(path[spike_times, 0], path[spike_times, 1], color=colors[i], edgecolor='none', s=3, alpha=1, zorder=1)

    ax_inset.text(50, 88, f'Place Cells: {valid_pcs}', ha='center', fontsize=label_fs, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    place_cell_ids2 = all_metrics_mnr['MnR_removed'][example2plot_2]['full_place_cell_ids']
    valid_pcs2 = len(place_cell_ids2)
    top_cell_ids2 = place_cell_ids2[:20]
    if len(top_cell_ids2) > 0:
        colors2 = plt.cm.turbo(np.linspace(0., 1., len(top_cell_ids2)))
        for i, cell_id in enumerate(top_cell_ids2):
            cell_mask2 = spikes2[:, 1] == cell_id
            pc_spikes2 = spikes2[cell_mask2, 0].astype(int)
            spike_times2 = pc_spikes2[(pc_spikes2 >= STARTDEL) & (pc_spikes2 < len(path2))]
            if len(spike_times2) > 0:
               ax_inset2.scatter(path2[spike_times2, 0], path2[spike_times2, 1], color=colors2[i], edgecolor='none', s=3, alpha=1, zorder=1)

    ax_inset2.text(50, 88, f'Place Cells: {valid_pcs2}', ha='center', fontsize=label_fs, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax_inset.set_box_aspect(1)
ax_inset.set_xlim(0, 100)
ax_inset.set_ylim(0, 100)
ax_inset.set_xticklabels([])
ax_inset.set_yticklabels([])
ax_inset.tick_params(axis='both', which='both', length=0)
ax_inset.set_title('Example Path: Ctr.', fontsize=title_fs, pad=0)

ax_inset2.set_box_aspect(1)
ax_inset2.set_xlim(0, 100)
ax_inset2.set_ylim(0, 100)
ax_inset2.set_xticklabels([])
ax_inset2.set_yticklabels([])
ax_inset2.tick_params(axis='both', which='both', length=0)
ax_inset2.set_title('Example Path: −MnR', fontsize=title_fs, pad=0)

# General legend over Panel C / D
legend_handles = [
    mpatches.Patch(facecolor=to_rgba('gray', alpha=1.0), edgecolor='black', hatch='', label='Ctr'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='−MnR'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='−MnR→\nVIP/CR\n-IS-3'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='xxxx', label='−MnR→\nVIP/CCK-BC\n& VIP-NVM')
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.68, 0.99), ncol=4, frameon=False, fontsize=tick_fs, handlelength=2.0, handleheight=2.0, columnspacing=2.0, handletextpad=0.4)

os.makedirs('Plots', exist_ok=True)
plot_name = 'Plots/plot_MnR_removal_place_cells'
print(f"Saving to {plot_name}.png and .pdf ...")
plt.savefig(plot_name + '.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(plot_name + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

print('Complete.')

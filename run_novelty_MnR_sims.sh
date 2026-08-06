#!/bin/bash

# --- CONFIGURATION ---
BASE_RUN=14
BASE_TRIAL=14

RUN_0=$((BASE_RUN + 0))

# --- DYNAMIC DIRS LIST GENERATION ---
DIRS=(
  "Simulation_Results/2D_box_Familiar2Novelty/Control/Trial_${BASE_TRIAL}/Run_${RUN_0}"
  "Simulation_Results/2D_box_Familiar2Novelty/Silenced_MnR/Trial_${BASE_TRIAL}/Run_${RUN_0}"
)

echo "--------------------------------------------------------"
echo "PRE-RUN DIRECTORY CHECK:"
# --- STATUS REPORT & DELETION PROMPT ---
for d in "${DIRS[@]}"; do
    if [ -d "$d" ]; then
        echo "[EXISTS]: $d"
        echo "--------------------------------------------------------"
        echo "Contents: $(ls -A1 $d | tr '\n' ' ')"

        read -p "Directory exists. Delete this specific folder? (y/n) " -n 1 -r
        echo # Move to a new line
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$d"
            echo "  Deleted: $d"
        else
            echo "Operation cancelled. Exiting to protect existing data."
            exit 1
        fi
    else
        echo "[CLEAR]:  $d (does not exist yet)"
    fi
done
echo "--------------------------------------------------------"
echo "All paths clear. Proceeding to generation..."

export NRN_PYLIB=/Users/agmccrei/opt/anaconda3/envs/lfpy/lib/libpython3.7m.dylib

# NEURON embeds Python. Ensure its embedded interpreter can find the full
# standard library from the conda env.
PYTHONHOME="/Users/agmccrei/opt/anaconda3/envs/lfpy"
PYTHONPATH="/Users/agmccrei/opt/anaconda3/envs/lfpy/lib/python3.7/site-packages"

# Wrapper function so we don't repeat the env var everywhere
run_sim() {
  conda run --no-capture-output -n lfpy env NRN_PYLIB="$NRN_PYLIB" \
    PYTHONHOME="$PYTHONHOME" PYTHONPATH="$PYTHONPATH" \
    ./x86_64/special -nogui "$@" Network_2D_box.hoc
}

echo "Generating paths and inputs for Novelty condition..."
cd make_inputs_linear_track
python make_grid_like_inputs_2D_box.py $RUN_0 4
cd ..

# --- Condition 0: Novelty (Control - MnR active) ---
echo "--------------------------------------------------------"
echo "Running Novelty condition with MnR active (Control)..."
run_sim -c n_runs=$RUN_0 -c n_trials=$BASE_TRIAL -c n_neuron=0 -c learning_cond=4

# --- Condition 0: Novelty (Silenced MnR) ---
echo "--------------------------------------------------------"
echo "Running Novelty condition with MnR silenced (Silenced_MnR)..."
run_sim -c n_runs=$RUN_0 -c n_trials=$BASE_TRIAL -c n_neuron=9 -c learning_cond=4

echo "All done!"

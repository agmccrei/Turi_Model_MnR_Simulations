# Running 2D Box Familiar-Novelty Conditioning Simulations

This guide explains how to run the NEURON simulations for the familiar to novelty contexts in the **2D Box Environment**:

The circuit is implemented in a single `Network_2D_box.hoc` file.

### Step 1: Compile mod files with the command:

nrnivmodl mechanisms/

### Step 2: Run the control and MnR removal simulations (each command simulates 1 random run). Note: edit the bash files to the correct python path directories for your system.
bash run_novelty_MnR_sims.sh
bash run_novelty_MnR_pathway_blocks.sh

### Step 3: Analyze and plot
python plot_MnR_removal_place_cells.py

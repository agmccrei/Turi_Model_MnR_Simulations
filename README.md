# Running 2D Box Familiar-Novelty Conditioning Simulations

Note: All code in this repository is adapted from the CA1 circuit model simulations published in Turi et al., 2019: 
Turi, G. F., Li, W. K., Chavlis, S., Pandi, I., O’Hare, J., Priestley, J. B., Grosmark, A. D., Liao, Z., Ladow, M., Zhang, J. F., Zemelman, B. V., Poirazi, P., & Losonczy, A. (2019). Vasoactive intestinal polypeptide-expressing interneurons in the hippocampus support goal-oriented spatial learning. Neuron, 101(6), 1150-1165.

This guide explains how to run the NEURON simulations for the familiar to novelty contexts in the **2D Box Environment**:

The circuit is implemented in a single `Network_2D_box.hoc` file.

### Step 1: Compile mod files with the command:

nrnivmodl mechanisms/

### Step 2: Run the control and MnR removal simulations (each command simulates 1 random run). Note: edit the bash files to the correct python path directories for your system.
bash run_novelty_MnR_sims.sh
bash run_novelty_MnR_pathway_blocks.sh

### Step 3: Analyze and plot
python plot_MnR_removal_place_cells.py

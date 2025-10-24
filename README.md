# hierarchy_band_grid

Content: This repository contains the code and analysis scripts for our paper ["Unfolding the Black Box of Recurrent Neural Networks for Path Integration"](https://neurips.cc/virtual/2025/poster/117611) NeurIPS (2025). This is built upon the [code framework](https://github.com/ganguli-lab/grid-pattern-formation/tree/master) of [Sorscher et al. NeurIPS (2019)](https://proceedings.neurips.cc/paper/2019/hash/6e7d5d259be7bf56ed79029c4e621f44-Abstract.html). Documentation updates in progress for the upcoming conference.

## 🗂️ Directory Structure
```
hierarchy_band_grid/
├── models/                               # Store data of different models
├── neural_circuit/                       # Neural Circuit
├── 1_model_genetate_data.ipynb           # Data generation
├── 2_band_cell_plot.ipynb                # Band cell plotting
├── 3_band_isomap.ipynb                   # Band isomap
├── 4_prune_input.ipynb                   # Prune input
├── 5_prune_read_out.ipynb                # Prune read-out
├── 6_prune_recurrent.ipynb               # Prune recurrent
├── 7_prune_cell_type.ipynb               # Prune cell type
├── environment.yaml                      # Environment dependencies
├── LICENSE                               # License file
├── main.py                               # Main entry point
├── model.py                              # Model definitions
├── place_cells.py                        # Place cell-related code
├── pruner.py                             # RNN Pruner 
├── README.md                             # Project description file
├── scores.py                             # Scoring script
├── trainer.py                            # Training script
├── trajectory_generator.py               # Trajectory generator
├── utils.py                              # Utility functions
├── visualize.py                          # Visualization functions
```

## ⚙️ Installation
To replicate the computational environment required to run this code:

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/yuling-wu/hierarchy_band_grid.git
    cd hierarchy_band_grid
    ```

2.  **Create the Conda environment** from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate PI_GC
    ```

## 💻 Usage
1. **Train a model:** 

    To train your own models, configure the parameters according to your needs and run the training script:
    ```bash
    python main.py
    ```
    **Note**: Depending on the RNN architecture, choice of training hyperparameters, and available computing resources, training can take several hours to complete.


2. **Quick Start with Pre-trained Models**

    For immediate analysis without training, use the provided example files, which are copied from [code framework](https://github.com/ganguli-lab/grid-pattern-formation/tree/master):

    - `models/example_pc_centers.npy`: Pre-computed place cell centers
    - `models/example_trained_weights.npy`: Pre-trained model weights for Vanilla RNN.

3. **Interactive Analysis with Jupyter Notebook**

    Launch Jupyter Notebook to explore and execute the analysis pipelines:

    - `1_model_generate_data.ipynb` for calculating the activations, band score, and grid score of all cells; and the spacing, orientation, phase, and direction of band cells.
    - `2_band_cell_plot.ipynb` for analyzing the main properties of band cells (Figure 4).
    - `3_band_isomap.ipynb` for visualizing the 3D Isomap of the population activities of band cells (Figure 4f).
    - `4_prune_input.ipynb`, `5_prune_read_out.ipynb`, `6_prune_recurrent.ipynb`, and `7_prune_cell_type.ipynb` for performing pruning experiments (Figure 3).

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. This license allows for reuse and modification with appropriate attribution.


## 📧 Contact

For questions regarding the code and analysis, please open an issue on GitHub or contact:
-   Tianhao Chu - chutianhao@stu.pku.edu.cn
-   Yuling Wu - yulingwu@stu.pku.edu.cn

Please feel free to submit pull requests, report bugs, or suggest new features.

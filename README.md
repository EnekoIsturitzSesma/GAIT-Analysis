# Deep Learning for Multi-Pathological Gait Event Detection using Minimal Wearable IMUs

This repository contains the code, models and experiments developed for the automatic segmentation of gait events (*Gait Event Detection*: starting and finnishing timestep for the swing phase of each foot) for subjects ranging from healthy to severely pathological walking patterns.

This work proposes an end-to-end **unified Deep Learning model (CNN + BiLSTM)** trained over *RAW* signals, it is optimized for data-scarce configurations (with less sensors) and it is capable of generalizing across multiple clinical conditions, as oposed to more traditional signal processing techniques.
---

## Key Aspects

1. **Multipathology Unified Model:** One unique DL model trained over **260 subjects** (1356 trials), covering healthy subjects (`HS`) and 7 other pathologies both neurologic (`CVA`, `PD`, `CIPN`, `RIL`) and orthopaedic (`HOA`, `KOA`, `ACL`).
2. **Minimized Sensor Usage:**Systematic study of sensor configurations, obtaining comparable `f1-scores` for the full 4 sensor dataset and reduced versions of 1, 2 or 3 sensors.

---
## Dataset

Link: https://doi.org/10.6084/m9.figshare.28806086

Voisard, C., Barrois, R., l’Escalopier, N.d. et al. A Dataset of Clinical Gait Signals with Wearable Sensors from Healthy, Neurological, and Orthopedic Cohorts. Sci Data 12, 1674 (2025). https://doi.org/10.1038/s41597-025-05959-w
---

## Repository Structure

```text
├── datasets/GAIT_signals/dataset       # Default dataset path for scripts and notebooks
├── models/   
│   └── LSTMGait.py                     # Model architecture definition (a small standard LSTM, and the CNN+BiLSTM model)
├── notebooks/                          # Jupyter notebooks for validation and experimentation
│   ├── ablation_datasize.ipynb         # Ablation study of the performance of the model across increasing subsets of the training data
│   ├── data_configuration.ipynb        # Search for the best data configuration for the model (window size, stride, ...)
│   ├── model_refination.ipynb          # Search for the best model configuration and hyperparameters for the problem
│   ├── pruebas.ipynb                   
│   └── sensor_test.ipynb               # Experiment testing the performance of the model across different sensor configurations
├── src/                                # Python scripts that provide essential functions for the experiments
│   ├── load_data_gait.py               # Script providing data loading functionality
│   ├── plot_data_gait.py               # Script providing data plotting functionality
│   ├── train_LSTMGait.py               # Script providing model training functionality
└── README.md



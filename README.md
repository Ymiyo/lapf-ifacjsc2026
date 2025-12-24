# Language-Aided Particle Filter

Code for numerical experiments in the IFAC WC 2026 submission "Language-Aided State Estimation".

## Run (minimal)

Minimal commands to reproduce the main experiment (LAPF vs EDAPF).

```bash
cd lapf-ifacjsc2026

# 1) Train models
python -m src.lapf_project.training.train_lapf_nn
python -m src.lapf_project.training.train_edapf_nn

# 2) Run experiment
python -m src.experiments.lapf_vs_edapf.run_lapf_vs_edapf
```

## Setup

### Requirements

- Python 3.12
- (Optional) NVIDIA GPU + CUDA-enabled PyTorch

## How to run

### 1) Training

Train the LAPF classifier model:

```bash
cd lapf-ifacjsc2026
python -m src.lapf_project.training.train_lapf_nn --labels 5 --epochs 100 --batch-size 16 --lr 1e-5
```

Train the EDAPF prediction model:

```bash
python -m src.lapf_project.training.train_edapf_nn --epochs 100 --batch-size 16 --lr 1e-5
```

### 2) Experiments

Experiment 1: LAPF vs EDAPF

```bash
python -m src.experiments.lapf_vs_edapf.run_lapf_vs_edapf
```

Experiment 2: Out-of-domain (dialect) mixed setting

```bash
python -m src.experiments.lapf_vs_edapf_under_out_of_domain.run_lapf_vs_edapf_under_out_of_domain
```

Experiment 3: Ablation on the number of quantization labels (num_labels)

Train models with different --labels:

```bash
python -m src.lapf_project.training.train_lapf_nn --labels 2
python -m src.lapf_project.training.train_lapf_nn --labels 3
# ...
python -m src.lapf_project.training.train_lapf_nn --labels 10
```

Run the ablation script:

```bash
python -m src.experiments.ablation_num_labels.run_ablation_num_labels
```

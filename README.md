# Neural_ODE
DL 2025 team project. Reproducible comparison of continuous‑time Neural ODE models against discrete‑time baselines (ResNet, RNN, LSTM) on irregular time‑series forecasting tasks.


## ✅ Completed Tasks
1. Generated synthetic dataset: A toy linear ODE system.
2. Implemented `ODEFunc` class: simple linear transformation of hidden state.
3. Used `torchdiffeq.odeint` to solve the ODE: predicted state trajectories with initial value problem solver.
4. Training + visualisation


## 🔧 Remaining Tasks
- Implement and train RNN-based models: Compare with standard RNN, GRU, or LSTM architectures on the same task.
- Build and evaluate ODE-RNN model: Fuse ODE solver with RNN encoder as in the Latent ODE architecture.
- Compare performance: Run benchmarks and visualize error metrics vs time steps or noise.
- Use a richer or real-world dataset: e.g. irregular medical time series, financial time series.
- Organize project repository: Create structured GitHub repo with README.md, dataset loaders, training scripts.
- Prepare presentation slides: Explain motivation, theory, models, results, comparison, and conclusions.


| **Member / GitHub handle**            | **Primary Deliverables**          | **Detailed Weekly Tasks & Milestones**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **@member1 Dmitry – ResNet**     | *Image/sequence ResNet baseline*  | <br>• Review dataset specs; design ResNet depth/width grid.<br>• Stub `src/models/resnet.py`, forward pass + unit-test.<br><br>• Integrate Lightning Module, add early-stopping & LR-scheduler.<br>• Run first training sweep → baseline metrics on W\&B.<br><br>• Add configurable bottleneck / dilated blocks; ablation script.<br>• Deliver inference-latency profiler notebook.<br><br>• Final tune (≤1 % MSE target); export `resnet.ckpt`, update results table, write ½-page methods subsection.                      |
| **@member2 Artemiy – RNN (GRU)** | *GRU baseline for irregular Δt*   | • Implement `TimeAwareGRU` (Δt concatenation) in `src/models/rnn.py`.<br>• Unit-test hidden-state reset & masking.<br> • Training loop; log N\_par, FLOPs.<br>• Hyper-opt hidden-units vs seq-length; compare to ResNet.<br> • Document failure modes & lessons-learned slide.                                                                                                                                                                                                                                             |
| **@member3 Akmuhammed – LSTM**   | *Standard + Time-LSTM baseline*   |• Fork Artemiy’s dataloader; add `TimeLSTM` variant.<br>  • Run same sweeps; capture learning-curve SVGs.<br>• Implement teacher-forcing vs scheduled-sampling toggle.<br> • Provide extrapolation demo GIF + write comparative paragraph.                                                                                                                                                                                                                                                                                  |
| **@member4 Alina – ODE**         | *Neural ODE / Latent ODE module*  |  • Fork `torchdiffeq`; wrap `ODEBlock` + adjoint tests.<br> • Reproduce paper’s spiral demo; log NFE stats.<br> • Plug into shared trainer; tune solver tolerances vs error.<br>• Add memory-profiling callback.<br> • Stretch: continuous normalising-flow or CNF head.<br>• Prepare “ODE-internals” slide & code walkthrough.                                                                                                                                                                                              |
| **Anita – Data & MLOps**         | *Dataset, CI, sweeps, evaluation* | • Freeze dataset (data-card.md), write `dataloader.py`.<br>• Set up repo structure, pre-commit, Black/isort, GitHub Actions (lint + unit tests + smoke-train).<br> • Hydra/YAML config schema; W\&B project; Optuna sweep script.<br>• Provide `run.sh --model=<name>` wrapper.<br> • Evaluation notebook: metrics table, paired t-test, NFE curves.<br>• Collect all checkpoints; generate leaderboard CSV.<br>  • Assemble final figures, ensure reproducibility (`docker/`), polish README & deliver 10-min Colab demo. |



____________________________________________________________________
# README.MD draft

---

## ⭐ Project Highlights

* **≤ 1 % error target** (MSE) on chosen dataset
* Unified experiment engine (Hydra + PyTorch Lightning)
* End‑to‑end reproducibility: one‑command run, CI smoke test, W\&B tracking
* Extensible plug‑in interface — drop your `Model` class, rerun, compare

---

## 1. Quick Start

```bash
# clone
git clone https://github.com/rainbowbrained/Neural_ODE.git
cd Neural_ODE
# example run (Neural ODE)
python train.py model=ode dataset=physionet
```

*See `configs/` for all options.*

---

## 2. Dataset

| Name               | Domain               | Samples           | Δt Characteristics | License |
| ------------------ | -------------------- | ----------------- | ------------------ | ------- |
| PhysioNet ICU 2012 | Vital signs (health) | 8 k patients      | Highly irregular   | MIT     |
| Synthetic Spiral   | Toy                  | 10 k trajectories | Uniform            | MIT     |

Download will auto‑trigger on first run and cache under `data/`.

---

## 3. Directory Layout

```
├── data/             # raw & processed datasets
├── src/
│   ├── models/       # resnet.py, rnn.py, lstm.py, ode.py
│   ├── datamodules/  # pytorch‑lightning data modules
│   ├── utils/        # common helpers
│   └── train.py      # entry point
├── configs/          # Hydra YAML configs
├── sweeps/           # hyper‑parameter search definitions
├── tests/            # unit & integration tests
├── scripts/          # helper bash/nb scripts
├── results/          # artifacts & metrics (auto‑generated)
├── environment.yml   # conda env spec
└── LICENSE
```

---

## 4. Training & Evaluation

| Model          | Params | Train NFE | Val MSE ↓ | Inference latency (ms) |
| -------------- | ------ | --------- | --------- | ---------------------- |
| ResNet‑18      | 11 M   |  –        |     |                   |
| GRU            | 1.2 M  |  –        |     |                   |
| LSTM           | 1.3 M  |  –        |     |                   |
| **Neural ODE** | 0.8 M  | 57        |     |                   |

*Numbers are placeholders – see `results.csv` for latest.*

---

## 5. Team & Roles

| Member       | GitHub                | Responsibility                  |
| ------------ | --------------------- | ------------------------------- |
| @member1 Dmitry | ResNet Lead           | Deep residual baseline          |
| @member2 Artemiy | RNN Lead              | GRU baseline                    |
| @member3 Akmuhammed | LSTM Lead             | LSTM baseline                   |
| @member4 Alina | ODE Lead              | Neural ODE module               |
| @rainbowbrained  | Data & MLOps  | Dataset, CI, sweeps, evaluation |

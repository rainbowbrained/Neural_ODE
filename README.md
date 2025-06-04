# Neural ODE 🧩⏱️  
Reproducible continuous-time deep-learning benchmark. DL 2025 team project. Reproducible comparison of continuous‑time Neural ODE models against discrete‑time baselines (ResNet, RNN, LSTM) on irregular time‑series forecasting tasks.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org/) 

Neural ODE compares **continuous-depth models** against classic discrete-time baselines (ResNet-18, GRU, Time-LSTM) on **irregular time-series and vision**.  

## 📊 Results
| Task / Metric                | ResNet-18   | GRU (Δt) | Time-LSTM  | **Neural ODE** |
| ---------------------------- | ----------- | -------- | ---------- | -------------- |
| **MNIST** – Acc ↑            | **99.31 %** | 97.40 %  | 99.23 %    | 99.17 %        |
|   Params                     | 11.7 M      | 1.2 M    | 11.7 M     | **0.21 M**     |
|   Latency ↓                  | 11 ms       | 0.49 ms     | **3 ms**   | 3 ms           |
| **CIFAR-10** – Acc ↑         | **95.1 %**  | 48.70 %   | 94.8 %     | 74.2 %         |
|   Latency ↓                  | 12 ms       | 0.83 ms     | **4 ms**   | 201 ms         |
| **PhysioNet 2012** – AUROC ↑ | 0.742       | 0.786    | 0.693      | **0.754**      |
|   Latency ↓                  | 7 ms        | 1.69 ms   | **0.6 ms** | 2.0 ms         |


## 🗂️ Datasets
| Name                   | Domain                   | Samples            | Δt pattern           |
| ---------------------- | ------------------------ | ------------------ | -------------------- |
| **PhysioNet ICU 2012** | 41 vital-sign channels   | 8 k patients       | *Highly irregular*   |
| MNIST / CIFAR-10       | Vision (28×28 / 32×32×3) | 60 k / 50 k images | *Uniform*            |



##  🏗️ Repository Layout

```
├── 1806.07366v5.pdf            # main paper
│
├── DL_NeuralODE.pdf      
├── GRU_implementation.ipynb
├── LSTM_MNIST_CIFAR-10_PhysioNet.ipynb
│
├── NeuralODE.ipynb            # initial Neural ODE draft
├── odenet_cifar10_metric.py   # Neural ODE on CIFAR10
├── odenet_mnist_metric.py     # Neural ODE on MNIST
├── odenet_physionet.py        # Neural ODE on PhysioNet
│
├── ResNet_CIFAR.ipynb        # ResNet on CIFAR
├── ResNet_MNIST.ipynb        # ResNet on MNIST
│
├── utils.py          
└── report.pdf                # project defense presentation
```

---

## 5. Team & Roles

| Member       | GitHub                | Responsibility                  |
| ------------ | --------------------- | ------------------------------- |
| @dpetrov835 Dmitry | ResNet Lead           | Deep residual baseline          |
| @Sklaveman Artemiy | RNN Lead              | GRU baseline                    |
| @Akmuhammet01 Akmuhammed | LSTM Lead             | LSTM baseline                   |
| @alina2002200 Alina | ODE Lead              | Neural ODE module               |
| @rainbowbrained  | Data & MLOps  | Dataset, CI, sweeps, evaluation |

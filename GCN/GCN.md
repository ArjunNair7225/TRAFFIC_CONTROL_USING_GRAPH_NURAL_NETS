# PEMS-BAY Traffic Flow Forecasting using Graph Convolutional Networks (GCN)

This repository implements a **Graph Convolutional Network (GCN)** to forecast traffic speeds and detect network congestion patterns using the benchmark **PEMS-BAY** traffic dataset. 

Our experimental workflow tracks a clear architecture progression: moving from **Raw Unscaled Data** to **Feature Normalization**, and finally migrating from a **Static Distance Adjacency Matrix** to a **Dynamically Computed Correlation Sensor Graph** trained over an extended 50-epoch timeline.

---

## 📊 Final Performance Summary

| Model Variation | Data Scaling | Adjacency Matrix Source | Epochs | MAE (↓) | RMSE (↓) | MAPE (↓) | Test Accuracy | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw-Data GCN Baseline** | None | Official Static File (`adj_mx_bay.pkl`) | 15 | 8.6357 mph | 10.8277 mph | 16.05% | *N/A* | *N/A* |
| **2. Normalized Baseline GCN** | `StandardScaler` | Official Static File (`adj_mx_bay.pkl`) | 15 | 2.9999 mph | 5.2797 mph | 6.77%* | *N/A* | *N/A* |
| **3. Custom Graph GCN** | `StandardScaler` | Dynamic Pearson Matrix (r > 0.5) | 15 | 2.9206 mph | 5.1811 mph | 6.76% | *N/A* | *N/A* |
| **4. Extended Custom GCN (Final)** | **`StandardScaler`** | **Dynamic Pearson Matrix (r > 0.5)** | **50** | **2.9027 mph** | **5.1211 mph** | **6.69%** | **96.03%** | **0.7515** |

*\*Note: Initial raw mathematical MAPE evaluation on scaled data yielded an arithmetic anomaly of 998,652.56% due to division errors near 0.0 mph sensor velocities. This was resolved using a localized threshold filter evaluating speeds > 1.0 mph.*

---

## 🛠️ Dataset & Architecture Specifications

### Dataset Parameters (PEMS-BAY)
* **Dataset Shape:** 52,116 snapshots across 325 sensors
* **Operational Velocities:** Max Speed: 85.1 mph | Min Speed: 0.0 mph
* **Sequence Window Horizon:** 12 look-back timesteps (`SEQ_LEN = 12`)
* **Data Partitions:** 70% Train (570 batches) | 15% Val (123 batches) | 15% Test (123 batches)

### Model Layout & Parameters
The model leverages a 2-layer Graph Convolution network mapped directly to a fully-connected output layer.
GCN((gcn1): GCNLayer((W): Linear(in_features=12, out_features=64, bias=False))(gcn2): GCNLayer((W): Linear(in_features=64, out_features=64, bias=False))(out): Linear(in_features=64, out_features=1, bias=True))Total Trainable Parameters: 4,929Runtime Device: Cuda (NVIDIA T4 GPU)

---

## 📈 Experimental Progression & Loss Logs

### Phase 1: Raw-Data GCN Baseline (Unscaled)
Training natively on raw paramaters (0.0 ~ 85.1 mph) severely restricts backpropagation convergence because the inputs are mathematically too large for stable random weight initializations.
* **Epoch  1:** Train MSE: 171.9566 | Val MSE: 135.5968
* **Epoch  6:** Train MSE: 127.3352 | Val MSE: 129.8821
* **Epoch 15:** Train MSE: 113.7526 | Val MSE: 116.2660

### Phase 2: Feature Normalization Stage (`StandardScaler`)
Transforming feature parameters into a normal distribution (μ = 0.0, σ = 1.0) immediately stabilizes updates and drops errors.
* **Epoch  1:** Train MSE: 0.3950 | Val MSE: 0.4126
* **Epoch  6:** Train MSE: 0.3773 | Val MSE: 0.4086
* **Epoch 15:** Train MSE: 0.3745 | Val MSE: 0.4038

---

### 📉 Raw & Scaled Baseline Loss Curves
*Place your training progress graph here to show the dramatic difference normalization made:*

![Raw Data GCN Training Progress](path_to_your_raw_loss_plot.png)

---

### Phase 3: Transition to a Custom Adjacency Matrix
Instead of reading the static baseline file, we dynamically compute a customized spatial network by calculating Pearson correlation coefficients across speed profiles.
* **Graph Threshold Setup:** Only keep edges where positive correlation r > 0.5.
* **Graph Dimensions:** Generated 35,690 custom non-zero edges with an average node degree of 109.82.
* **Epoch  1:** Train MSE: 0.3605 | Val MSE: 0.3686
* **Epoch 15:** Train MSE: 0.3372 | Val MSE: 0.3612

### Phase 4: Extended Custom Graph Run (50 Epochs)
Extending the optimization run over 50 epochs allowed the GCN filters to completely adapt to custom spatial patterns, reaching maximum stability.
* **🏅 Epoch  1/50:** Train MSE: 0.3604 | Val MSE: 0.3689
* **🏅 Epoch 20/50:** Train MSE: 0.3370 | Val MSE: 0.3616
* **🏅 Epoch 40/50:** Train MSE: 0.3318 | Val MSE: 0.3547
* **🏅 Epoch 50/50:** Train MSE: 0.3298 | Val MSE: 0.3529

> **Optimal State Achieved:** Best Validation MSE reached **0.3529**; checkpoint successfully saved as `best_gcn_model.pth`.

---

## 📊 Final Performance Metrics Report

Evaluation parameters evaluated directly on the final unseen test split after mapping metrics back to true miles-per-hour (`mph`) space.

### 🔹 Regression Performances (Continuous Speed Values)
* **Mean Absolute Error (MAE):** 2.9027 mph
* **Root Mean Squared Error (RMSE):** 5.1211 mph
* **Mean Absolute Pct Error (MAPE):** 6.69%

### 🔹 Classification Performances (Congestion Detection)
To evaluate the model's physical utility in traffic control systems, velocities dropping below **50.0 mph** are flagged as **Congested (Class 1)**, while speeds at or above are treated as **Free Flow (Class 0)**.
* **Model Accuracy Score:** 96.03%
* **Model F1-Score:** 0.7515
* **Precision Score:** 0.7532
* **Recall (Sensitivity):** 0.7498

### 📝 Detailed Classification Breakdown
precision    recall  f1-score   supportFree Flow       0.98      0.98      0.98   2337194Congested       0.75      0.75      0.75    203331accuracy                           0.96   2540525macro avg       0.87      0.86      0.86   2540525weighted avg       0.96      0.96      0.96   2540525

### 🧠 Performance Insight Analysis
Traffic datasets are naturally highly imbalanced; notice there are **2,337,194** Free Flow records but only **203,331** Congested ones. Reaching an **F1-Score of 0.75** alongside an excellent balance of **0.7532 Precision** and **0.7498 Recall** proves that our dynamic Pearson matrix graph is accurately mapping spatial correlations and successfully capturing true physical bottlenecks instead of simply predicting the majority class.

---

## 🔮 Time-Series Forecasting Results
*Place your final prediction vs. actual speed snapshot graph here:*

![GCN Traffic Flow Prediction vs Actual - Sensor 0](/Assets/GCN.png)

---

## 💾 Instant Pre-trained Weight Loading

To load our saved checkpoint (`best_gcn_model.pth`) directly for instant forecasting evaluations without running the full training script again, use this PyTorch script:

```python
import torch

# Instantiate architecture framework
model = GCN(num_nodes=325, seq_len=12, hidden=64).to(device)

# Load best checkpoint weights
model.load_state_dict(torch.load('best_gcn_model.pth', map_location=device))
model.eval()

print("✔ Optimized GCN pre-trained model successfully loaded into system memory!")
```
# PEMS-BAY Traffic Speed Forecasting using Long Short-Term Memory (LSTM)

This repository implements a **Long Short-Term Memory (LSTM)** neural network to forecast traffic speeds across the sensors in the **PEMS-BAY** traffic dataset.

The workflow starts from the raw traffic-speed observations, applies **feature normalization**, converts the continuous time series into fixed-length sequences, and trains an LSTM model to predict the traffic speed of all 325 sensors at the next timestep.

---

##  Project Overview

Traffic speed is a time-dependent signal. An LSTM is used here because it is designed to learn patterns from sequential data and use information from previous timesteps to predict future values.

### Main workflow

```text
PEMS-BAY Traffic Data
        ↓
Traffic Speed Matrix
        ↓
Train / Validation / Test Split
        ↓
StandardScaler Normalization
        ↓
12-Timestep Sliding Window
        ↓
LSTM Model
        ↓
Next-Timestep Speed Prediction
        ↓
Inverse Scaling
        ↓
MAE / RMSE Evaluation
        ↓
Actual vs Predicted Visualization
```

---

##  Dataset

The project uses the **PEMS-BAY** traffic-speed dataset.

### Dataset Specifications

| Parameter | Value |
| :--- | :--- |
| Dataset | PEMS-BAY |
| Total time snapshots | **52,116** |
| Number of sensors | **325** |
| Time interval | **5 minutes** |
| Time period | **January 1, 2017 – June 30, 2017** |
| Original speed matrix | **(52,116, 325)** |
| Sequence length | **12 timesteps** |
| Forecast horizon | **Next timestep** |

Each row represents a timestamp and each column represents a traffic sensor.

The original data contains 325 sensor locations with corresponding traffic-speed measurements.

---

##  Data Preprocessing

### 1. Train / Validation / Test Split

The time series is divided chronologically so that future observations are not used to train the model.

| Split | Original Shape |
| :--- | :--- |
| Train | **(36,481, 325)** |
| Validation | **(7,817, 325)** |
| Test | **(7,818, 325)** |

The split corresponds approximately to:

- **70% Training**
- **15% Validation**
- **15% Testing**

---

### 2. Feature Scaling

Traffic speeds are normalized using `StandardScaler`.

The scaler is fitted on the training data and then applied to the validation and test data using the same transformation.

This produces normalized inputs with approximately:

```text
Mean (μ) ≈ 0
Standard deviation (σ) ≈ 1
```

The scaled datasets are:

```text
Scaled train      : (36,481, 325)
Scaled validation : (7,817, 325)
Scaled test       : (7,818, 325)
```

---

##  Sequence Generation

The model uses a **12-timestep look-back window**.

For every prediction:

```text
Previous 12 timesteps
        ↓
      LSTM
        ↓
Next timestep prediction
```

The sequence-generation process produces:

| Dataset | X Shape | y Shape |
| :--- | :--- | :--- |
| Train | **(36,469, 12, 325)** | **(36,469, 325)** |
| Validation | **(7,805, 12, 325)** | **(7,805, 325)** |
| Test | **(7,806, 12, 325)** | **(7,806, 325)** |

Therefore, each input sample contains:

- **12 historical timesteps**
- **325 sensor values per timestep**

and the target contains:

- **325 sensor values for the next timestep**

---

##  LSTM Model Architecture

The LSTM is configured with:

| Parameter | Value |
| :--- | :--- |
| Input size | **325** |
| Hidden size | **64** |
| Output size | **325** |
| Sequence length | **12** |
| Loss function | **MSELoss** |
| Optimizer | **Adam** |
| Learning rate | **0.001** |
| Gradient clipping | **1.0** |
| Epochs | **20** |
| Batch size | **64** |
| Device | **CUDA – NVIDIA GeForce RTX 4060 Laptop GPU** |

The model receives the traffic speeds of all 325 sensors over the previous 12 timesteps and produces the predicted speed for all 325 sensors at the next timestep.

---

## ️ Training Configuration

The training uses:

```text
Loss       : Mean Squared Error (MSE)
Optimizer  : Adam
Learning Rate : 0.001
Batch Size : 64
Epochs     : 20
Gradient Clipping : max_norm = 1.0
```

Gradient clipping is used to prevent excessively large gradients during LSTM training.

The best model checkpoint is saved as:

```text
best_lstm_model.pth
```

The checkpoint is selected based on the lowest validation loss.

---

##  Training Progress

| Epoch | Train Loss | Validation Loss |
| :---: | ---: | ---: |
| 1 | 0.4263 | 0.4336 |
| 2 | 0.3152 | 0.3907 |
| 3 | 0.2830 | 0.3682 |
| 4 | 0.2622 | 0.3566 |
| 5 | 0.2456 | 0.3415 |
| 6 | 0.2316 | 0.3308 |
| 7 | 0.2199 | 0.3195 |
| 8 | 0.2107 | 0.3163 |
| 9 | 0.2038 | 0.3154 |
| 10 | 0.1968 | 0.3047 |
| 11 | 0.1896 | 0.2994 |
| 12 | 0.1846 | 0.2968 |
| 13 | 0.1807 | 0.2965 |
| 14 | 0.1774 | 0.2852 |
| 15 | 0.1737 | 0.2845 |
| 16 | 0.1706 | 0.2862 |
| 17 | 0.1674 | 0.2767 |
| 18 | 0.1655 | 0.2788 |
| 19 | 0.1631 | 0.2773 |
| **20** | **0.1617** | **0.2750** |

### Training observation

The training loss decreases from **0.4263** to **0.1617**, while the validation loss decreases from **0.4336** to **0.2750**.

The lowest reported validation loss occurs at **Epoch 20**.

---

##  Final Performance Summary

The final model is evaluated on the unseen test split after predictions are mapped back to the original traffic-speed scale.

| Metric | LSTM Result |
| :--- | ---: |
| **MAE** | **2.5305 mph** |
| **RMSE** | **4.2249 mph** |
| Test predictions shape | **(7,806, 325)** |
| Actual values shape | **(7,806, 325)** |

### What the metrics mean

**MAE (Mean Absolute Error)**

```text
MAE = average(|Actual - Predicted|)
```

The model has an average absolute prediction error of approximately:

> **2.53 mph**

**RMSE (Root Mean Squared Error)**

```text
RMSE = √(average((Actual - Predicted)²))
```

The reported RMSE is:

> **4.22 mph**

RMSE gives greater weight to larger prediction errors.

---

##  Actual vs Predicted Traffic Speed

The following graph compares the actual and predicted traffic speed for **Sensor 0** over the first **200 test time steps**.

![LSTM Traffic Speed Prediction - Sensor 0](./lstm_prediction_sensor0.png)

### Graph Interpretation

The predicted curve generally follows the overall movement of the actual traffic-speed curve.

In the earlier portion of the sequence, both curves remain around the same traffic-speed range. Later, when the actual traffic speed decreases, the prediction also follows the downward trend, although some point-to-point differences remain.

This indicates that the LSTM is capturing the major temporal pattern of the selected sensor while still producing prediction errors during rapid changes.

---

##  Model Evaluation Pipeline

After training, the best checkpoint is loaded:

```text
best_lstm_model.pth
        ↓
Load trained weights
        ↓
Set model to evaluation mode
        ↓
Predict test set
        ↓
Compare predictions with actual values
        ↓
Calculate MAE and RMSE
```

The final prediction and actual arrays have the following shape:

```text
Predictions : (7806, 325)
Actual      : (7806, 325)
```

This means the model produces a prediction for **325 sensors at each test timestep**.

---

##  Hardware / Runtime

The notebook detected and used:

```text
CUDA Running Status : True
GPU : NVIDIA GeForce RTX 4060 Laptop GPU
```

The LSTM training therefore runs on the available CUDA GPU.

---

##  Loading the Pre-trained Model

The saved checkpoint can be loaded for evaluation or forecasting without retraining the model.

```python
model = LSTMModel(
    input_size=325,
    hidden_size=64,
    output_size=325
).to(device)

model.load_state_dict(
    torch.load(
        "best_lstm_model.pth",
        map_location=device
    )
)

model.eval()

print("Best LSTM model loaded successfully.")
```

---

##  Expected Project Files

A simple repository structure can be:

```text
.
├── main.ipynb
├── best_lstm_model.pth
├── lstm_prediction_sensor0.png
└── README.md
```

---

##  Reproducibility

The main experimental settings used in the notebook are:

```text
Dataset             : PEMS-BAY
Sensors             : 325
Sequence Length     : 12
Hidden Size         : 64
Batch Size          : 64
Epochs              : 20
Optimizer           : Adam
Learning Rate       : 0.001
Loss                : MSE
Gradient Clipping   : 1.0
```

---

##  Notes

- The model performs **multi-sensor traffic-speed forecasting**, predicting all 325 sensor values for the next timestep.
- The input sequence contains the previous **12 timesteps**, corresponding to **60 minutes of historical observations** because the data is sampled every 5 minutes.
- The reported MAE and RMSE are calculated on the test predictions after evaluation.
- The supplied experiment reports **MAE and RMSE**; no MAPE, accuracy, or F1-score result is included for the LSTM experiment.
- The prediction visualization shown in this README is for **Sensor 0** and the first **200 test time steps**.

---

##  Summary

The LSTM experiment demonstrates a sequence-based approach for forecasting traffic speeds in the PEMS-BAY network.

### Final Results

```text
              LSTM
               │
        ┌──────┴──────┐
        │             │
      MAE           RMSE
     2.5305         4.2249
      mph             mph
```

**Best validation loss:** `0.2750`

**Test MAE:** `2.5305 mph`

**Test RMSE:** `4.2249 mph`

**Model checkpoint:** `best_lstm_model.pth`

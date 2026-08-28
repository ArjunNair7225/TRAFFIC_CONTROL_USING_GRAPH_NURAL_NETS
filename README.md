Traffic Flow Prediction Using Graph Neural Networks

Project Overview

This project focuses on predicting traffic flow using Deep Learning and Graph Neural Networks (GNNs).

The main idea is to use traffic speed data collected from multiple road sensors and learn the relationship between different sensors. Since roads are connected to each other, they can be represented as a graph where sensors are treated as nodes and roads/connections are treated as edges.

The project will initially use an LSTM model as a baseline and later extend it to a GCN-LSTM based model.

Objectives

Predict future traffic speed using historical traffic data.

Learn temporal traffic patterns using Deep Learning.

Represent the road network as a graph.

Use Graph Neural Networks to learn relationships between traffic sensors.

Improve traffic flow prediction.

Use the predictions to help identify possible traffic congestion and support accident prevention.

Dataset

The project uses the PEMS-BAY traffic dataset.

The dataset contains traffic speed measurements collected from 325 traffic sensors in the Bay Area.

Dataset Information

Number of sensors: 325

Time interval: 5 minutes

Number of time steps: 52,116

Main feature: Traffic speed

File format: HDF5 (.h5)

The dataset is not included in this repository because of its size.

Data Representation

The traffic speed data is represented as:

Time Steps × Sensors
52116 × 325

Each row represents one timestamp and each column represents a traffic sensor.

Input and Output

The model uses previous traffic speeds to predict the next traffic speed.

Currently, we use a sequence length of 12. Since each measurement is taken every 5 minutes:

12 × 5 minutes = 60 minutes

Therefore:

Input:
Previous 12 time steps (1 hour)
        ↓
      Model
        ↓
Output:
Next 1 time step (5 minutes)

For all 325 sensors:

X shape = 12 × 325
y shape = 325

Data Splitting

The data is divided chronologically into:

70% Training data

15% Validation data

15% Testing data

The data is not randomly shuffled because traffic data is time-series data.

Past ------------------------------------> Future

|--------------|-----------|-------------|
   Training     Validation      Testing
      70%          15%           15%

Models

1. LSTM

Long Short-Term Memory (LSTM) is used as the initial baseline model.

It learns temporal patterns in traffic speed and predicts the next traffic condition from previous observations.

2. GCN

Graph Convolutional Network (GCN) will be used to learn relationships between different traffic sensors.

Traffic sensors will be represented as nodes in a graph.

3. GCN-LSTM

The final model will combine:

GCN → Spatial relationships
LSTM → Temporal relationships

Expected architecture:

Traffic Data
     ↓
Graph Construction
     ↓
GCN
     ↓
Spatial Features
     ↓
LSTM
     ↓
Future Traffic Speed

Technologies Used

Python

NumPy

Pandas

PyTorch

Scikit-learn

h5py

Matplotlib

Jupyter Notebook

Git & GitHub

Evaluation Metrics

The models will be evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Mean Absolute Percentage Error (MAPE)

The predicted traffic speed will be compared with the actual traffic speed.

Project Structure

TRAFFIC_CONTROL_USING_GRAPH_NEURAL_NETS/
│
├── main.ipynb
├── main.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
└── Data/
    └── pems-bay.h5

The dataset files are excluded from GitHub using .gitignore.

Current Progress

Project setup

Python environment setup

PEMS-BAY dataset obtained

HDF5 file explored

Traffic speed data extracted

Sensor IDs and timestamps identified

Input sequences created

Training/validation/testing split created

Data normalization

LSTM implementation

LSTM training

LSTM evaluation

Graph construction

GCN implementation

GCN-LSTM implementation

Final evaluation

Traffic congestion/accident-risk analysis

Future Scope

The final system can be extended to:

Predict traffic congestion.

Identify high-risk traffic conditions.

Provide early warnings for abnormal traffic patterns.

Integrate real-time traffic data.

Develop a traffic monitoring dashboard.

Use additional factors such as weather, road conditions and accidents.

Team

This is a group project developed as part of an academic project on traffic flow prediction using Deep Learning and Graph Neural Networks.

License

This project is intended for academic and educational purposes.
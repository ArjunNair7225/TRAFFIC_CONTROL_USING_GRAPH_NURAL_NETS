# Integrated GCN--LSTM Model: Training Results and Analysis

## 1. Overview

The integrated graph-convolutional LSTM model was trained on the
PEMS-BAY traffic-speed dataset. It combines spatial relationships among
traffic sensors with temporal dependencies in traffic speed.

-   **Dataset:** PEMS-BAY
-   **Sensors:** 325
-   **Input sequence:** 12 historical timesteps
-   **Sampling interval:** 5 minutes
-   **Forecast horizon:** Next timestep (5 minutes ahead)
-   **Graph:** Pearson-correlation adjacency, retaining positive
    correlations greater than 0.5
-   **Model:** Graph-convolutional recurrent LSTM
-   **Checkpoint:** `best_gcn_lstm_model.pth`

The model was trained successfully and generated test-set predictions.
However, the results do not yet establish that the integrated model is
better overall than a simple persistence baseline.

## 2. Test Results

  Model                              MAE (mph) ↓   RMSE (mph) ↓      MAPE (%) ↓
  -------------------------------- ------------- -------------- ---------------
  Persistence baseline                **0.9951**     **1.9942**      **1.9001**
  Integrated GCN--LSTM                    2.3693         4.2969          5.3008
  Standalone LSTM (previous run)          2.5305         4.2249   Not available
  Standalone GCN (previous run)           2.9027         5.1211            6.69

**Metric interpretation:** lower values are better. The standalone LSTM
and GCN results are historical results from earlier experiments. They
have not yet been verified under identical splits and preprocessing, so
the comparisons are indicative rather than definitive.

## 3. Is the Integrated Model Better?

### Compared with the historical standalone GCN

The integrated model's MAE is approximately **18.4% lower**, and its
RMSE is approximately **16.1% lower** than the previously reported GCN
result.

This is encouraging, but a fair conclusion requires rerunning the
standalone model with the same chronological split, scaler, graph
construction, windowing and evaluation code.

### Compared with the historical standalone LSTM

The integrated model's MAE is approximately **6.4% lower**, but its RMSE
is approximately **1.7% higher**.

Therefore, it does not outperform the historical LSTM on both metrics.
The models must be rerun under matching conditions before drawing a
definitive conclusion.

### Compared with persistence

The persistence baseline predicts the next speed using the latest
observed speed. Its MAE is **0.9951 mph** and RMSE is **1.9942 mph**.

The integrated model has approximately **2.38 times the MAE** and **2.15
times the RMSE** of persistence. This is the main issue to investigate
before claiming that the integrated model improves forecasting.

## 4. Training Diagnostics

  Diagnostic                       Result
  ----------------------- ---------------
  Final training MSE              0.19628
  Best validation MSE             0.27079
  Epochs completed                     50
  Training time             33.82 minutes
  Best checkpoint epoch                50

The training and validation losses generally declined, and the best
validation checkpoint was saved. There is a gap between training and
validation loss, indicating some generalization gap, but the reported
curves do not by themselves establish severe overfitting.

Because validation loss was still improving at epoch 50, a longer
training budget could be tested. Any change should be selected using
validation performance, not repeated tuning against the test set.

## 5. Congestion Classification

The reported congestion-classification results were:

-   **Accuracy:** 96.92%
-   **Congested-class F1-score:** 0.8088

These results are useful, but accuracy alone may be misleading when
free-flow observations are more common than congested observations.
Precision, recall, F1-score and the confusion matrix should be reported
together.

Classification results do not replace the speed-regression metrics MAE
and RMSE.

## 6. Important Checks Before Claiming Improvement

1.  **Verify target alignment.** Confirm that each prediction is
    compared with the correct next 5-minute observation.
2.  **Check units and inverse scaling.** Predictions and targets must
    both be inverse-transformed to mph before calculating final metrics.
3.  **Use identical test periods.** All models must evaluate the same
    timestamps and sensors.
4.  **Use identical preprocessing.** Keep the same chronological split,
    training-only scaler fitting, window generation and missing-value
    treatment.
5.  **Verify the persistence baseline.** Ensure it predicts the next
    speed using the last observed speed from each corresponding input
    window.
6.  **Rerun standalone models fairly.** Re-evaluate GCN and LSTM with
    the integrated notebook's split, preprocessing and metric functions.
7.  **Preserve the test set.** Use validation data for architecture and
    hyperparameter selection. Avoid repeatedly tuning against test
    results.

## 7. Recommended Next Steps

### Step 1 --- Keep the current checkpoint

Retain `best_gcn_lstm_model.pth` as the first integrated-model
experiment. Preserve the notebook, configuration and metrics used to
produce it.

### Step 2 --- Establish fair baselines

Rerun the standalone GCN and LSTM using the exact same data split,
preprocessing, prediction horizon and evaluation code as the integrated
model.

### Step 3 --- Investigate the persistence gap

Inspect the time alignment, scaler inversion, graph construction and
output predictions. Compare prediction errors across free-flow and
congested periods, as well as across individual sensors.

### Step 4 --- Run controlled experiments

After validating the pipeline, vary one factor at a time:

-   Pearson graph versus distance-based graph
-   Correlation threshold
-   Hidden size and learning rate
-   Sequence length
-   Fixed graph setup versus the research paper's Adjustable Hops Policy
-   Other architectural choices supported by the paper

Do not assume any change will improve the model; retain changes only
when validation and final controlled evaluation support them.

## 8. Research Status and Conclusion

The integrated GCN--LSTM model has been implemented and trained
successfully. It achieved **MAE = 2.3693 mph**, **RMSE = 4.2969 mph**
and **MAPE = 5.3008%** in the reported test evaluation.

Its MAE is lower than the historical standalone GCN and LSTM MAE values,
but its RMSE is slightly higher than the historical LSTM RMSE. Those
older runs require a controlled rerun before making a fair head-to-head
claim.

Most importantly, the persistence baseline performs substantially better
on the reported test metrics. The next research priority is to validate
the evaluation pipeline and establish matched baselines, then
investigate model improvements.

The current integrated model should be described as a **successfully
trained research baseline, not yet a demonstrated improvement over
persistence**. The WEST Adjustable Hops Policy remains a separate future
extension unless it has been explicitly implemented and evaluated.

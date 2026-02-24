# 📈 Multi Horizon Stock Forecasting

Deep learning based multi-horizon stock prediction system using:

- Random Forest (Baseline)
- LSTM
- Temporal Fusion Transformer (TFT)
- Prophet (Statistical Model)

---

## 🔍 Project Overview

This project predicts short-term stock movement (1-day ahead) using:

- Technical Indicators (RSI, MACD, EMA, SMA)
- Volatility Metrics
- Sentiment Features
- Market Index (VIX)

The goal is to compare classical ML, deep learning, and transformer-based models.

---

## 🏗️ Project Structure
src/
│
├── block_a_data_processing.py
├── block_b_feature_engineering.py
├── block_c_sentiment.py
├── block_d_dataset_merge.py
├── block_e_baseline_modeling.py
├── block_f_tft_model.py


---

## 📊 Models Implemented

### 1️⃣ Random Forest
Baseline ML model for regression.

### 2️⃣ LSTM
Sequential deep learning model for time series.

### 3️⃣ Temporal Fusion Transformer (TFT)
State-of-the-art transformer model for multi-horizon forecasting.

---
📈 Evaluation Metrics

MAE

RMSE

MAPE

📌 Future Improvements

Multi-horizon prediction (1D, 5D, 10D)

Hyperparameter tuning

Walk-forward validation

Model ensembling

Deployment using Streamlit
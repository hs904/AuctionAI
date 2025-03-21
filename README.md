# AuctionAI
# 🧠 Bidoo Auction Analytics: Predicting Winner Expenses

This repository presents a complete pipeline for predicting winner expenses in closed online auctions on the Bidoo platform. It compares traditional regression models with a deep machine learning approach (Neural Network), supported by detailed EDA, preprocessing, model training, and evaluation.

---

## 📁 Project Structure

```bash
├── Data/
│   ├── Raw_Data/closed_auctions.csv     # Original dataset
│   ├── prepared_data.csv                # Cleaned and feature-engineered dataset
│   ├── X_test.npy, y_test.npy           # Saved standardized test sets
│
├── Model/
│   ├── trained_model.pth                # Trained PyTorch NN model
│   ├── loss_history.npy                 # Training loss history
│   ├── OLS_Regression.pkl               # Trained OLS model
│   ├── Ridge_Regression.pkl             # Trained Ridge model
│
├── Evaluation_plots/
│   ├── model_comparison.csv             # Comparison metrics (OLS, Ridge, NN)
│   ├── *.png                            # Visualization plots
│
├── model_training.py                   # PyTorch Neural Network model
├── traditional_model.py                # Linear models training + comparison
├── model_evaluation.py                 # NN performance visualization
├── EDA.ipynb or script                 # Full exploratory data analysis
└── README.md                           # This file
```

---

## 🔍 1. Exploratory Data Analysis (EDA)

- Loaded and inspected auction-level data with 49,523 observations.
- No missing values detected.
- Categorical features like `name` and `winner_modality` were analyzed.
- `name` was categorized into broader item categories (e.g., Electronics, Gift Cards).
- Log transformation was applied to skewed numerical features.
- Correlation heatmap and modality behavior across categories were plotted.

---

## 🛠️ 2. Feature Engineering & Preprocessing

- Removed unnecessary columns (e.g., timestamp, usernames).
- Created new features:
  - `price_per_bid`
  - `bid_intensity`
  - `shipping_ratio`
- Log-transformed numerical columns to improve model performance.
- Encoded categorical variables.

---

## 🤖 3. Model Training

###  `model_training.py`

- Defines a **deep neural network** with:
  - 3 hidden layers
  - Dropout regularization
  - Xavier initialization
  - Adam optimizer with LR scheduler + early stopping
- Trained over 300 epochs with batch size 64.
- Early stooping to avoid overfitting.
- Saves model and training loss history.

---

## 📉 4. Traditional Models & Comparison

###  `traditional_model.py`

- Trains:
  - OLS Regression
  - Ridge Regression
- Loads neural network model
- Evaluates all 3 models (OLS, Ridge, NN) using:
  - MAE, MSE, RMSE, R²
- Saves model performance and bar charts to `Evaluation_plots/`

📈 **Model Performance:**

```
                  MAE      MSE      RMSE     R² Score
OLS Regression     0.383   0.308    0.555    0.741
Ridge Regression   0.383   0.308    0.555    0.741
Neural Network     0.064   0.023    0.153    0.980
```

📌 Neural Network performs best in terms of MAE, MSE, and R².

---

## 🧪 5. NN Model Evaluation

###  `model_evaluation.py`

- Loads saved PyTorch model
- Generates:
  - Residual distribution
  - Predicted vs. actual scatter plot
  - Training loss curve

Confirms good fit and learning convergence.

---

## 📌 Conclusion

This project demonstrates a robust pipeline from EDA to modeling:

- Carefully engineered features and normalized distributions.
- Traditional and deep models benchmarked.
- Neural network outperforms OLS and Ridge regressions.

## 🚀 How to Run

```bash
# Step 1: Preprocess and generate prepared_data.csv
python preprocessing.py

# Step 2: Train neural network
python model_training.py

# Step 3: Train traditional models and compare
python traditional_model.py

# Step 4: Evaluate NN model visually
python model_evaluation.py
```

---


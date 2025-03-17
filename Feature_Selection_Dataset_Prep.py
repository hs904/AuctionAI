import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "closed_auctions.csv" 
df = pd.read_csv(file_path)

# Step 1: Drop Unnecessary Columns
df.drop(columns=["currency", "timestamp", "winner_username"], inplace=True, errors="ignore")
print("Remaining Columns:", df.columns)

# Step 2: Encode Categorical Variables

# Encode 'item_category' 
encoder = LabelEncoder()
df["item_category_encoded"] = encoder.fit_transform(df["item_category"])
# Convert 'winner_modality' to numeric (if not already done)
df["winner_modality"] = df["winner_modality"].map({"manual": 0, "automatic": 1})
# Drop original 'item_category' column after encoding
df.drop(columns=["item_category"], inplace=True)

# Step 3: Log Transformation
# Apply log transformation to skewed numerical features
df["log_bid"] = np.log1p(df["bid"])
df["log_price"] = np.log1p(df["price"])
df["log_winner_expenses"] = np.log1p(df["winner_expenses"])
# Drop original highly skewed columns
df.drop(columns=["bid", "price", "winner_expenses"], inplace=True)


# Verify the updated dataset
print(df.head())

# Step 4: Feature Selection
# Define features (X) and target (y)
features = ["log_bid", "num_bids", "timer", "log_price", "log_winner_expenses", 
            "winner_modality", "item_category_encoded"]
target = "win_outcome"  # Binary target (1 = Win, 0 = Lose)

X = df[features]
y = df[target]

# Verify the shape of features and target
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

#Step 5: Train-Test Split
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify dataset split
print("Training Set:", X_train.shape, y_train.shape)
print("Testing Set:", X_test.shape, y_test.shape)

# Step 6: Standardize Numerical Features
from sklearn.preprocessing import StandardScaler

# Define which features to scale
numeric_features = ["log_bid", "num_bids", "timer", "log_price", "log_winner_expenses"]

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data and transform both train & test sets
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Verify scaling
print(X_train.head())


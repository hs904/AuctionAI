import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "./Data/Raw_Data/closed_auctions.csv" 
df = pd.read_csv(file_path)

# Step 1: Drop Unnecessary Columns
df.drop(columns=["currency", "timestamp", "winner_username"], inplace=True, errors="ignore")
print("Remaining Columns:", df.columns)

# Step 2: Encode Categorical Variables

# Categorizing the "name" column into meaningful item categories
def categorize_item(name):
    name = name.lower()
    if "bidoo" in name or "points" in name:
        return "Bidoo Credits"
    elif any(x in name for x in ["€"]):
        return "Gift Card"
    elif any(x in name for x in ["iphone", "samsung", "xiaomi", "airpods", "macbook", "tablet", "laptop", "tv", "camera","earbuds"]):
        return "Electronics"
    elif any(x in name for x in ["dyson", "nespresso", "vacuum", "blender", "kitchen", "coffee", "fridge", "oven"]):
        return "Home Appliances"
    elif any(x in name for x in ["gucci", "rolex", "louis vuitton", "luxury", "prada", "chanel", "hermes"]):
        return "Luxury Goods"
    elif any(x in name for x in ["playstation", "nintendo", "xbox", "gaming", "console"]):
        return "Gaming & Entertainment"
    elif any(x in name for x in ["scooter", "bike", "car", "vehicle", "motorcycle"]):
        return "Transportation"
    else:
        return "Other"

# Apply item categorization
df["item_category"] = df["name"].apply(categorize_item)

# Drop the original 'name' column after categorization
df.drop(columns=["name"], inplace=True)

# Encode 'item_category' 
encoder = LabelEncoder()
df["item_category_encoded"] = encoder.fit_transform(df["item_category"])

# Convert 'winner_modality' to numeric
df["winner_modality"] = df["winner_modality"].map({"manual": 0, "automatic": 1})

# Drop original 'item_category' column after encoding
df.drop(columns=["item_category"], inplace=True)


# Step 3: Log Transformation
# Apply log transformation to skewed numerical features
df["log_bid"] = np.log1p(df["bid"])
df["log_price"] = np.log1p(df["price"])
df["log_winner_expenses"] = np.log1p(df["winner_expenses"])

# Create a new column - "Additonal cost to win"
df["additional_cost_to_win"] = df["winner_expenses"] - df["bid"]

# Drop original highly skewed columns
df.drop(columns=["bid", "price", "winner_expenses"], inplace=True)

# Save the processed dataset as a new CSV file
prepared_csv_path = "./Data/prepared_data.csv"
df.to_csv(prepared_csv_path, index=False)

#################################################
# Step 4: Define Features (X) and Target (y)
'''
Target Variable (y)
Since this is a bidding fee auction (penny auction), the primary goal should be to predict 
the "additional cost to win" (the cost incurred by the user to win the auction). 
This is more meaningful than simply predicting the probability of winning or the final
bidding price, especially given the dynamics of penny auctions, where the outcome is 
influenced by factors like bidding timing, competition, and sunk-cost fallacy.

Expected additional cost to win (additional_cost_to_win)
Defined as: additional cost = winner expenses - user's bid
This tells us how much more a user might need to bid to win.
'''
# Define Target Variable (y)
y = df["additional_cost_to_win"]

# Define features (X)
features = ["num_bids", "shipping_expenses", "timer", "winner_modality", 
            "winner_num_bids", "item_category_encoded", "log_bid", "log_price", "log_winner_expenses"]

X = df[features]

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save into csv
X_train.to_csv('./Data/X_train.csv', index=False)
y_train.to_csv('./Data/y_train.csv', index=False)
X_test.to_csv('./Data/X_test.csv', index=False)
y_test.to_csv('./Data/y_test.csv', index=False)

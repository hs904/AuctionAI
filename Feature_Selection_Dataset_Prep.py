import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "./Data/Raw_Data/closed_auctions.csv" 
df = pd.read_csv(file_path)

# Step 1: Drop Unnecessary Columns
df.drop(columns=["currency", "timestamp", "winner_username","winner_num_bids"], inplace=True, errors="ignore")
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

# Step 3: Feature Engineering
# Log Transformation
# Apply log transformation to skewed numerical features
df["log_bid"] = np.log1p(df["bid"])
df["log_price"] = np.log1p(df["price"])
df["log_winner_expenses"] = np.log1p(df["winner_expenses"])

# Add New Features
df["price_per_bid"] = df["log_price"] / df["num_bids"]  # Price per bid
df["bid_intensity"] = df["num_bids"] / df["timer"]  # Bidding frequency (higher means aggressive bidding)
df["shipping_ratio"] = df["shipping_expenses"] / df["log_winner_expenses"]  # Shipping cost as % of total cost

# Replace infinite values resulting from division errors
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Drop original highly skewed columns
df.drop(columns=["bid", "price","winner_expenses"], inplace=True)

# Save the processed dataset as a new CSV file
prepared_csv_path = "./Data/prepared_data.csv"
df.to_csv(prepared_csv_path, index=False)
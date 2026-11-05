import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Install: pip install pandas numpy scikit-learn
df = pd.read_csv("raw_inventory_logistics.csv")
print("Raw shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().copy()
numeric = ["Quantity_Received", "Quantity_Dispatched", "Unit_Cost",
           "Supplier_Lead_Time_Days", "Transport_Distance_KM"]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Convert impossible values to missing, then impute with median
for c in ["Quantity_Received", "Quantity_Dispatched", "Transport_Distance_KM"]:
    df.loc[df[c] < 0, c] = np.nan
df.loc[df["Unit_Cost"] <= 0, "Unit_Cost"] = np.nan
df.loc[df["Supplier_Lead_Time_Days"] <= 0, "Supplier_Lead_Time_Days"] = np.nan
for c in numeric:
    df[c] = df[c].fillna(df[c].median())

# Detect and cap extreme values with the IQR rule
for c in numeric:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    df[c] = df[c].clip(lower, upper)

# Create useful logistics features
df["Inventory_Flow_Units"] = df["Quantity_Received"] + df["Quantity_Dispatched"]
df["Estimated_Dispatch_Value"] = df["Quantity_Dispatched"] * df["Unit_Cost"]

# Scale numeric features (retain original cleaned columns)
df[[c+"_minmax" for c in numeric]] = MinMaxScaler().fit_transform(df[numeric])
df[[c+"_zscore" for c in numeric]] = StandardScaler().fit_transform(df[numeric])
df.to_csv("cleaned_inventory_logistics.csv", index=False)
print("Cleaned shape:", df.shape)
print("Saved cleaned_inventory_logistics.csv")

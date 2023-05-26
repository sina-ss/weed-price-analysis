import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
data = pd.read_csv("weed_clean.csv")
regulations = pd.read_csv("regulations.csv")
merged_data = pd.merge(data, regulations, on="state")

plt.figure(figsize=(10, 6))
plt.scatter(
    data["per_capita_income"],
    data["highQ_price_ounce"],
    color="blue",
    label="High Quality",
)
plt.scatter(
    data["per_capita_income"],
    data["medQ_price_ounce"],
    color="green",
    label="Medium Quality",
)
plt.xlabel("Per Capita Income")
plt.ylabel("Price of Weed (per ounce)")
plt.title("Weed Price vs Per Capita Income")
plt.legend()
plt.show()

# 2
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.hist(data["highQ_price_ounce"], bins=20, color="blue", edgecolor="black")
plt.title("Distribution of High Quality Weed Prices")
plt.xlabel("Price per Ounce ($)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(data["medQ_price_ounce"], bins=20, color="green", edgecolor="black")
plt.title("Distribution of Medium Quality Weed Prices")
plt.xlabel("Price per Ounce ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(12, 12))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# 4
plt.figure(figsize=(10, 6))
sns.boxplot(x="legality", y="highQ_price_ounce", data=merged_data)
plt.title("High Quality Weed Price vs Legality", fontsize=16)
plt.xlabel("Legality")
plt.xticks(rotation=45)
plt.ylabel("High Quality Weed Price (per ounce)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="legality", y="medQ_price_ounce", data=merged_data)
plt.title("Medium Quality Weed Price vs Legality", fontsize=16)
plt.xlabel("Legality")
plt.xticks(rotation=45)
plt.ylabel("Medium Quality Weed Price (per ounce)")
plt.show()

# 5
plt.figure(figsize=(10, 6))
sns.boxplot(x="legality", y="percent_black", data=merged_data)
plt.title("Percentage of Black Population vs Legality of Weed")
plt.xticks(rotation=90)
plt.show()

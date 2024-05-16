import pandas as pd

#1
# Print name and class section
print("Addison Zheng, CSCI 4371")

#2
# Read provided data file into variable
fpath = 'Assignment 1/ml_a1_data.csv'
data = pd.read_csv(fpath)

#3
# Calculate and print the number of observations
print("Number of observations:", len(data))

#4
# Calculate and print the number of variables
LotArea = data["LotArea"]
SalePrice = data["SalePrice"]
print("Number of Variables:", LotArea.count() + SalePrice.count())

#5
# Calculate and print min, median, mean, max, and std
print("\nLotArea minimum value:", LotArea.min())
print("LotArea median value:", LotArea.median())
print("LotArea mean value:", LotArea.mean())
print("LotArea max value:", LotArea.max())
print("LotArea std value:", LotArea.std())

print("\nSalePrice minimum value:", SalePrice.min())
print("SalePrice median value:", SalePrice.median())
print("SalePrice mean value:", SalePrice.mean())
print("SalePrice max value:", SalePrice.max())
print("SalePrice std value:", SalePrice.std())
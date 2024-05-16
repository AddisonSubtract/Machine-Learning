import pandas as pd

fpath = 'Project 1/Cleaned Data 2.csv'
data = pd.read_csv(fpath)

#3
# Calculate and print the number of observations
print("Number of observations:", len(data))

#4
# Calculate and print the number of variables
CriticScore = data["Critic_Score"]

#5
# Calculate and print min, median, mean, max, and std
print("\nLotArea minimum value:", CriticScore.min())
print("LotArea mean value:", CriticScore.mean())
print("LotArea max value:", CriticScore.max())
print("LotArea std value:", CriticScore.std())

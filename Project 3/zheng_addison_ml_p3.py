import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
import numpy as np
from sklearn import preprocessing

fpath = 'Project 3/Cleaned Data 2.csv'
data = pd.read_csv(fpath)

X = data["Global_Sales"].values.reshape(-1, 1)
y = data["Critic_Score"]

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

warnings.filterwarnings("ignore", message="The least populated class in y has only 1 members, which is less than n_splits=5.")

knn = KNeighborsClassifier()
origScores = cross_val_score(knn, X, y, cv=5)
print("Original Cross-validation scores:", origScores)
print("Original Mean Scores:", np.mean(origScores))

knn10 = KNeighborsClassifier(n_neighbors=10)
Score10 = cross_val_score(knn10, X, y, cv=5)
print("K-Value 10 Cross-validation scores:", Score10)
print("K-Value 10 Mean Scores:", np.mean(Score10))

knn50 = KNeighborsClassifier(n_neighbors=50)
Score50 = cross_val_score(knn50, X, y, cv=5)
print("K-Value 50 Cross-validation scores:", Score50)
print("K-Value 50 Mean Scores:", np.mean(Score50))

knn100 = KNeighborsClassifier(n_neighbors=100)
Score100 = cross_val_score(knn100, X, y, cv=5)
print("K-Value 100 Cross-validation scores:", Score100)
print("K-Value 100 Mean Scores:", np.mean(Score100))

knnManhattan = KNeighborsClassifier(p=1)
ScoreManhattan = cross_val_score(knnManhattan, X, y, cv=5)
print("P-Value Manhattan Cross-validation scores:", ScoreManhattan)
print("P-Value Manhattan Mean Scores:", np.mean(ScoreManhattan))
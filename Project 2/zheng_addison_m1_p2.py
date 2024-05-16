import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

fpath = 'Project 2/Cleaned Data 2.csv'
data = pd.read_csv(fpath)

X_train = data["Global_Sales"].values.reshape(-1, 1)
y_train = data["Critic_Score"]

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes= 200).fit(X_train, y_train)
origScores = cross_val_score(regr, X_train, y_train, cv=5)
print("Original Cross-validation scores:", origScores)
print("Original Mean Scores:", np.mean(origScores))

identity = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes= 200, activation='identity').fit(X_train, y_train)

identityScores = cross_val_score(identity, X_train, y_train, cv=5)
print("Identity Cross-validation scores:", identityScores)
print("Identity Mean Scores:", np.mean(identityScores))

logistic = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes= 200, activation='logistic').fit(X_train, y_train)

logisticScores = cross_val_score(logistic, X_train, y_train, cv=5)
print("Logistic Cross-validation scores:", logisticScores)
print("Logistic Mean Scores:", np.mean(logisticScores))

tanh = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes= 200, activation='tanh').fit(X_train, y_train)

tanhScores = cross_val_score(tanh, X_train, y_train, cv=5)
print("Tanh Cross-validation scores:", tanhScores)
print("Tanh Mean Scores:", np.mean(tanhScores))

# param_grid = {
#     'activation': ['identity']
# }

# grid_search = GridSearchCV(regr, param_grid, cv=5, n_jobs=-1)

# grid_search.fit(X_train, y_train)

# print("Parameter:", grid_search.best_params)
# bestmodel = grid_search.best_estimator
# scores = grid_search.best_score
# print("Best Cross-validation Score (Mean Squared Error):", scores)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# 1. Read in the dataset from assignment 1: ml_a1_data.csv
fpath = 'Assignment 2/ml_a1_data.csv'
data = pd.read_csv(fpath)

# 2. Split the dataset into a training set
df = data[['LotArea', 'SalePrice']]

x = df["LotArea"].values.reshape(-1, 1)
y = df["SalePrice"].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# 3. Make a scatterplot of the training set and save it
plt.figure(figsize=(10, 6))
plt.scatter(x_train,y_train)
plt.title("Lot Area By Sale Price")
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.show()

# 4. Perform linear regression
regr = linear_model.LinearRegression()

# 4a. Fit a linear regression model to the training set
regr.fit(x_train,y_train)
y_pred_train = regr.predict(x_train)
# 4ai. The R2 score
r2_train = r2_score(y_train, y_pred_train)
print(f"Linear Regression training R^2: {r2_train:.4f}")
# 4aii. The root mean squared error
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Linear Regression training root mean squared error: {rmse_train:.4f}")

# 4c. Plot the model line with the scatterplot of the training set and save
t0, t1 = regr.intercept_[0], regr.coef_[0][0]

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train)
plt.plot(x_train, t0 + t1 * x_train, c='red')
plt.title("Linear Lot Area By Sale Price")
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')

plt.show()

# 4d.	Evaluate the model on the test set
regr.fit(x_test,y_test)
y_pred_test = regr.predict(x_test)
# 4di. The R2 score
r2_test = r2_score(y_test, y_pred_test)
print(f"Linear Regression test R^2: {r2_test:.4f}")
# 4dii. The root mean squared error
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Linear Regression test root mean squared error: {rmse_test:.4f}")

# 5a. Perform polynomial regression
poly = PolynomialFeatures(5)
X_train_poly = poly.fit_transform(x_train)
X_test_poly = poly.transform(x_test)

# 5b. Fit a linear regression model to X_train_poly and y_train.
regr.fit(X_train_poly, y_train)
poly_y_train_pred = regr.predict(X_train_poly)

# 5c. Evaluate the model on the training set 
# 5ci. The R2 score
poly_r2_train = r2_score(y_train, poly_y_train_pred)
print(f"Polynomial Regression training R^2: {poly_r2_train:.4f}")
# 5cii. The root mean squared error
poly_rmse_train = np.sqrt(mean_squared_error(y_train, poly_y_train_pred))
print(f"Polynomial Regression training root mean squared error: {poly_rmse_train:.4f}")

# 5d. Plot the model line with the scatterplot of the training set
x_train_sorted, y_train_pred_sorted = zip(*sorted(zip(x_train.flatten(), poly_y_train_pred.flatten())))

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train)
plt.plot(x_train_sorted, y_train_pred_sorted, c='red')
plt.title("Polynomial Regression: Lot Area By Sale Price")
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.show()

# 5e. Evaluate the model on the training set 
poly_y_test_pred = regr.predict(X_test_poly)
# 5ei. The R2 score
poly_r2_test = r2_score(y_test, poly_y_test_pred)
print(f"Polynomial Regression test R^2: {poly_r2_test:.4f}")
# 5eii. The root mean squared error
poly_rmse_test = np.sqrt(mean_squared_error(y_test, poly_y_test_pred))
print(f"Polynomial Regression test root mean squared error: {poly_rmse_test:.4f}")

# 6. Perform polynomial regression with regularization
# 6i Ridge regression
clf = Ridge(alpha=1.0)
# 6b. Fit the chosen model to X_train_poly and y_train 
clf.fit(X_train_poly, y_train)

# 6c. Evaluate the model on the training set 
ridge_poly_y_train_pred = clf.predict(X_train_poly)
# 6ci. The R2 score
ridge_poly_r2_train = r2_score(y_train, ridge_poly_y_train_pred)
print(f"Polynomial Regression with regularization training R^2: {ridge_poly_r2_train:.4f}")
# 6cii. The root mean squared error
ridge_poly_rmse_train = np.sqrt(mean_squared_error(y_train, ridge_poly_y_train_pred))
print(f"Polynomial Regression with regularization training root mean squared error: {ridge_poly_rmse_train:.4f}")

# 6d. Plot the model line with the scatterplot of the training set
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train)
plt.plot(x_train_sorted, ridge_poly_y_train_pred, c='red')
plt.title("Polynomial Regression with Regularization: Lot Area By Sale Price")
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.show()

# 6e. Evaluate the model on the training set 
ridge_poly_y_test_pred = regr.predict(X_test_poly)
# 6ei. The R2 score
ridge_poly_r2_test = r2_score(y_test, ridge_poly_y_test_pred)
print(f"Polynomial Regression with regularization test R^2: {ridge_poly_r2_test:.4f}")
# 6eii. The root mean squared error
ridge_poly_rmse_test = np.sqrt(mean_squared_error(y_test, ridge_poly_y_test_pred))
print(f"Polynomial Regression with regularization test root mean squared error: {ridge_poly_rmse_test:.4f}")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

#1. Import the iris dataset and split it 
iris = load_iris(as_frame=True)
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

logReg = LogisticRegression(max_iter=150)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn15 = KNeighborsClassifier(n_neighbors=15)

#2. Use 5-fold cross validation with the training data
logRegScores = cross_val_score(logReg, X_train, y_train)
knn5Scores = cross_val_score(knn5, X_train, y_train)
knn15Scores = cross_val_score(knn15, X_train, y_train)

#3. Print the 5 validation scores for each of the models
print("Logistic Regression Validation Scores:", logRegScores)
print("KNN5 Validation Scores:", knn5Scores)
print("KNN15 Validation Scores:", knn15Scores)

#4. Print the mean validation score for each of the models.
print("Mean Logistic Regression Validation Score:", np.mean(logRegScores))
print("Mean KNN5 Validation Scores:", np.mean(knn5Scores))
print("Mean KNN15 Validation Scores:", np.mean(knn15Scores))

#5
print("Best model should be Logistic Regression")

#6
#a. Create a new model instance
newModel = LogisticRegression(max_iter=150)
#b. Train this model on the training set without using cross-validation
newModel.fit(X_train, y_train)
#c. Use the test set to get a score for the model
score = newModel.score(X_test, y_test)
#d. Use the test set to generate a confusion matrix.
y_pred = newModel.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
#e. Print the score and the confusion matrix
print("Score:", score)
print("Matrix:\n", matrix)
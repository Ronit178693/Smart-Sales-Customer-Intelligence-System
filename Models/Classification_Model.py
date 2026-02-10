import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
# Importing processed data directly
from Data_Preprocessing import train_x, test_x, train_y, test_y
import joblib

# Training Logistic Regression
LR = LogisticRegression()

#  Using Cross-Valadition 
cv_scores = cross_val_score(
    LR,
    train_x,
    train_y['Churn_Label'],
    cv=5,
    scoring="accuracy"
)
print("Cross-Validation Scores are: ", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
#  Our model performs well

#Performing Randomized Search CV
#These are the hyperparameters we are going to tune
param_dist = {
    "C": loguniform(1e-4, 1e2),
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "class_weight": [None, "balanced"]
}

random_search = RandomizedSearchCV(
    estimator=LR,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

random_search.fit(train_x, train_y['Churn_Label'])

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Using the best model from RandomizedSearch
best_model = random_search.best_estimator_
best_model.fit(train_x, train_y['Churn_Label'])

# Predictions and Evaluation
pred_y = best_model.predict(test_x)
print("Classification Report:")
print(classification_report(test_y['Churn_Label'], pred_y))

print("Confusion Matrix:")
print(confusion_matrix(test_y['Churn_Label'], pred_y))
# Our model is working perfectly 

joblib.dump(best_model, 'pkl/Classification_Model.pkl')
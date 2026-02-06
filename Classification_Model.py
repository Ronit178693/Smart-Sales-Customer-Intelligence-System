import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# Importing processed data directly
from Data_Preprocessing import train_x, test_x, train_y, test_y

# Training Logistic Regression
LR = LogisticRegression()
LR.fit(train_x, train_y['Churn_Label'])

# Predictions and Evaluation
pred_y = LR.predict(test_x)
print("Classification Report:")
print(classification_report(test_y['Churn_Label'], pred_y))

print("Confusion Matrix:")
print(confusion_matrix(test_y['Churn_Label'], pred_y))

# Our model is working perfectly 

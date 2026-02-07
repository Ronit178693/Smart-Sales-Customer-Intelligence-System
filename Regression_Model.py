import numpy as np
import pandas as pd
from Data_Preprocessing import train_x, test_x, train_y, test_y
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


df = pd.read_csv("preprocessed_data.csv")


Linear = LinearRegression()
cross_val = cross_val_score(
    Linear,
    train_x,
    train_y['Next_Month_Spend_Label'],
    cv=5,
    scoring="r2"
)

print("Cross Validation Scores: ", cross_val)
print("Mean Cross Validation Score: ", np.mean(cross_val))
# Model is performing good while Cross Validating

Linear.fit(train_x, train_y['Next_Month_Spend_Label'])

pred_y = Linear.predict(test_x)
    
# Regression model evaluation
print("Mean Absolute Error: ", mean_absolute_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Mean Squared Error: ", mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y)))
print("R-squared Score: ", r2_score(test_y['Next_Month_Spend_Label'], pred_y))
#Linear Regression works well for this dataset
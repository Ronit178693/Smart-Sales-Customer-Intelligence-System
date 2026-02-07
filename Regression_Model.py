import numpy as np
import pandas as pd
from Data_Preprocessing import train_x, test_x, train_y, test_y
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("preprocessed_data.csv")


Linear = LinearRegression()
Linear.fit(train_x, train_y['Next_Month_Spend_Label'])

pred_y = Linear.predict(test_x)
    
# Regression model evaluation
print("Mean Absolute Error: ", mean_absolute_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Mean Squared Error: ", mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y)))
print("R-squared Score: ", r2_score(test_y['Next_Month_Spend_Label'], pred_y))
#Linear Regression works well for this dataset
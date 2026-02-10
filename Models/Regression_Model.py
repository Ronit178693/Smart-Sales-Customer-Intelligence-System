import numpy as np
import pandas as pd
from Data_Preprocessing import train_x, test_x, train_y, test_y
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import joblib

# We don't need to read the csv again as we are importing the data from Data_Preprocessing
# df = pd.read_csv("preprocessed_data.csv") 


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

# Hyperparameter Tuning for Linear Regression
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=Linear,
    param_distributions=param_grid,
    n_iter=4,
    cv=5,
    scoring='r2',
    random_state=42
)

random_search.fit(train_x, train_y['Next_Month_Spend_Label'])

print("Best Parameters:", random_search.best_params_)
print("Best R2 Score:", random_search.best_score_)

# Using the best model
best_model = random_search.best_estimator_
best_model.fit(train_x, train_y['Next_Month_Spend_Label'])

pred_y = best_model.predict(test_x)
    
# Regression model evaluation
print("Mean Absolute Error: ", mean_absolute_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Mean Squared Error: ", mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(test_y['Next_Month_Spend_Label'], pred_y)))
print("R-squared Score: ", r2_score(test_y['Next_Month_Spend_Label'], pred_y))
#Linear Regression works well for this dataset

joblib.dump(best_model, 'pkl/Regression_Model.pkl')
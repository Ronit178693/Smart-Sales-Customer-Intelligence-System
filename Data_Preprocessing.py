import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder




df = pd.read_csv("customer_data.csv")

# print(df.isnull().sum())# Null or missing values in each column
#To check if there are outliers present 
#As there are many features we are not going to visualize every feature so we are going to use stastical method
#Using the IQR Method
numeric_cols = df.select_dtypes(include='number').columns#Gets all the numeric columns from the dataset
for i in numeric_cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)] 
    #Getting only the features with outliers
    if not outliers.empty:
        print(f"Outliers for {i} are present: {outliers}")
        df.drop(outliers.index, inplace=True)
# Outliers are present in features Avg_Monthly_Spend, Last_Month_Spend and Next_Month_Spend_Label

# Performing Encoding
Label = LabelEncoder()
df['Gender'] = Label.fit_transform(df['Gender'])
OneHot = OneHotEncoder()
df['Location'] = OneHot.fit_transform(df['Location'])


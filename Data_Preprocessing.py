import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("customer_data.csv")

print(df.isnull().sum())# Null or missing values in each column

# Float columns are considered string so converting them to int 
df['Avg_Monthly_Spend'] = df['Avg_Monthly_Spend'].astype(int)
df['Last_Month_Spend'] = df['Last_Month_Spend'].astype(int)
df['Next_Month_Spend_Label'] = df['Next_Month_Spend_Label'].astype(int)
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

#OneHot Encoding
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

#Standardization the numeric columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Dropping the CustomerID column
# Performing minor Feature selection
df.drop('CustomerID', axis=1, inplace=True) # Making this change directly to the corrent dataframe
# Separating features (X) and targets (y)
target_cols = ['Churn_Label', 'Next_Month_Spend_Label']

# y contains the answers (labels) we need for training later
y = df[target_cols]

# X contains only the features for PCA to analyze
X = df.drop(target_cols, axis=1)

# Using PCA for feature extraction
pca = PCA(n_components = 0.95) # keeping 95% of the varience
df_pca = pca.fit_transform(X)
print(f"Original shape: {X.shape}, Reduced shape: {df_pca.shape}")
print(df_pca)




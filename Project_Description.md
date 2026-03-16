# Smart Sales Customer Intelligence System - Detailed Project Description

## 1. Project Overview
The **Smart Sales Customer Intelligence System** is a machine learning-based web application built using **Streamlit**. It is designed to help businesses understand customer behavior, predict churn risks, segment customers into distinct groups, and forecast future customer spending. The system acts as a comprehensive dashboard where sales and marketing teams can input customer data (either manually or by selecting from an existing dataset) and receive actionable intelligence.

## 2. Core Features
The application offers three main predictive capabilities based on trained machine learning models:
1.  **Customer Segmentation (Clustering)**: Groups customers into clusters (Group 0, 1, or 2) based on their profiles and purchasing behavior, allowing for targeted marketing strategies.
2.  **Churn Risk Prediction (Classification)**: Predicts whether a customer is at "High Risk" or "Low Risk" of leaving (churning), enabling proactive retention efforts.
3.  **Future Value Forecasting (Regression)**: Predicts the exact dollar amount a customer is likely to spend in the following month, helping in revenue forecasting.

## 3. Technology Stack
*   **Frontend & Web Framework**: Streamlit (`app.py`) for the interactive user interface.
*   **Data Manipulation**: Pandas, NumPy.
*   **Machine Learning**: Scikit-Learn (`sklearn`) for model building, preprocessing, and evaluation.
*   **Data Serialization**: `joblib` for saving and loading trained models and preprocessors.
*   **Visualization** (during training): Matplotlib, Seaborn.

## 4. System Architecture & Project Structure
*   **`app.py`**: The main application file containing the Streamlit UI, user input forms, logic to load models, data preprocessing steps for inference, and the final display of predictions.
*   **`Dataset/`**:
    *   `customer_data.csv`: The raw dataset containing existing customer profiles and historical data.
    *   `preprocessed_data.csv`: The cleaned and transformed dataset used for training the models.
*   **`Models/`**: Contains the scripts used to train the machine learning models.
    *   `Data_Preprocessing.py`: Handles data cleaning, feature engineering, and splitting data into training and testing sets.
    *   `Classification_Model.py`: Trains the Logistic Regression model for churn prediction.
    *   `Regression_Model.py`: Trains the Linear Regression model for predicting next month's spend.
    *   `Unsupervised_model.py`: Trains the KMeans clustering model using the Elbow pattern to find optimal segments.
*   **`pkl/`**: Stores the serialized machine learning artifacts required by the frontend application.
    *   `scaler.pkl`: StandardScaler to normalize numerical features.
    *   `pca.pkl`: Principal Component Analysis model for dimensionality reduction before passing data to models.
    *   `gender_encoder.pkl`: LabelEncoder for the 'Gender' categorical variable.
    *   `Classification_Model.pkl`: The trained Logistic Regression model.
    *   `Regression_Model.pkl`: The trained Linear Regression model.
    *   `kmeans_model.pkl`: The trained KMeans clustering model.

## 5. Machine Learning Models - Deep Dive
### Data Preprocessing (`Data_Preprocessing.py` & `app.py`)
Before feeding data into the models, the system processes 9 key features:
*   **Demographics**: Age, Gender, Location (Urban, Suburban, Rural). Categorical variables are encoded (Label Encoding for Gender, One-Hot Encoding for Location).
*   **Behavioral & Financial Data**: Tenure (Months), Avg Monthly Spend, Last Month Spend, Num Transactions, Days Since Last Purchase, Support Tickets.
*   All numerical features are standardized using `StandardScaler`.
*   Data undergoes Principal Component Analysis (PCA) to reduce dimensionality while capturing the most important variance in the dataset.

### A. Churn Classification (`Classification_Model.py`)
*   **Algorithm**: Logistic Regression.
*   **Optimization**: Hyperparameters (`C`, `penalty`, `solver`, `class_weight`) were tuned using `RandomizedSearchCV` with 5-fold Cross-Validation to find the best performing setup.
*   **Output**: Binary label deciding if a user is "High Risk" (1) or "Low Risk" (0).

### B. Spend Regression (`Regression_Model.py`)
*   **Algorithm**: Linear Regression.
*   **Optimization**: Hyperparameters (`fit_intercept`, `positive`) were tuned using `RandomizedSearchCV` to minimize errors like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
*   **Output**: A continuous numerical value representing the estimated next month's spend.

### C. Customer Segmentation (`Unsupervised_model.py`)
*   **Algorithm**: KMeans Clustering.
*   **Optimization**: The optimal number of clusters (k=3) was determined visually using the Elbow Method (plotting WCSS against the number of clusters) and evaluated using the Silhouette Score. 
*   **Output**: Assigns the user to one of 3 distinct clusters.

## 6. User Workflow in the App
1.  **Selection**: The user opens the dashboard and chooses between "Existing Customer" or "Manual Entry" from the sidebar.
2.  **Input**: 
    *   If *Existing Customer* is chosen, selecting a Customer ID auto-populates all input fields using `customer_data.csv`.
    *   If *Manual Entry* is chosen, the user manually fills in statistics like Age, Spend, Location, etc.
3.  **Inference**: Upon clicking "Predict":
    *   The backend transforms the inputs precisely as the training data was (encoding, scaling, PCA).
    *   The transformed array is passed to the three separate `.pkl` models simultaneously.
4.  **Results**: The dashboard displays the predicted Customer Segment (e.g., "Group 0"), Churn Risk status dynamically colored (Red for High Risk, Green for Low), and the exact predicted Future Value in dollars.

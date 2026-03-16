import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
st.set_page_config(page_title="Smart Sales Intelligence", layout="wide")

# --- Helper Functions ---
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects."""
    paths = {
        'scaler': 'pkl/scaler.pkl',
        'pca': 'pkl/pca.pkl',
        'kmeans': 'pkl/kmeans_model.pkl',
        'classification': 'pkl/Classification_Model.pkl',
        'regression': 'pkl/Regression_Model.pkl',
        'encoder': 'pkl/gender_encoder.pkl'
    }
    
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
            return None
    return models

@st.cache_data
def load_data():
    """Load the customer dataset."""
    if os.path.exists("Dataset/customer_data.csv"):
        return pd.read_csv("Dataset/customer_data.csv")
    return None

def preprocess_input(data, models):
    """
    Transform raw input data into the format expected by the models.
    """
    input_df = pd.DataFrame([data])
    
    # 1. Encode Gender
    try:
        # Use the encoder if available and label matches
        input_df['Gender'] = models['encoder'].transform(input_df['Gender'])[0]
    except:
        # Fallback
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # 2. Encode Location (One-Hot Encoding)
    # Logic: Rural -> 0,0 | Suburban -> 1,0 | Urban -> 0,1
    # We must ensure Location_Suburban and Location_Urban columns exist
    
    loc = input_df.pop('Location').values[0]
    input_df['Location_Suburban'] = 1 if loc == 'Suburban' else 0
    input_df['Location_Urban'] = 1 if loc == 'Urban' else 0
    
    # 3. Scale Numeric Features
    numeric_features = ['Age', 'Tenure_Months', 'Avg_Monthly_Spend', 'Last_Month_Spend', 
                        'Num_Transactions', 'Days_Since_Last_Purchase', 'Support_Tickets']
    
    input_df[numeric_features] = models['scaler'].transform(input_df[numeric_features])
    
    # 4. Reorder for PCA
    pca_cols = ['Age', 'Gender', 'Tenure_Months', 'Avg_Monthly_Spend', 'Last_Month_Spend',
                'Num_Transactions', 'Days_Since_Last_Purchase', 'Support_Tickets',
                'Location_Suburban', 'Location_Urban']
    
    input_df = input_df[pca_cols]
    
    # 5. PCA Transform
    pca_data = models['pca'].transform(input_df)
    
    return pca_data

# --- Main App ---
st.title("🛍️ Smart Sales Customer Intelligence")

# Load Resources
models = load_models()
df = load_data()

if models:
    # --- Sidebar ---
    st.sidebar.header("Input Configuration")
    
    # Input Mode Selection
    input_mode = st.sidebar.radio("Select Input Mode:", ["Manual Entry", "Existing Customer"])
    
    # Default Values
    defaults = {
        'Gender': "Male", 'Age': 30, 'Location': "Urban", 'Tenure': 12,
        'Avg_Spend': 500, 'Last_Spend': 450, 'Transactions': 5, 
        'Days_Since': 10, 'Tickets': 1
    }
    
    selected_customer_id = None
    
    if input_mode == "Existing Customer":
        if df is not None:
            customer_ids = df['CustomerID'].tolist()
            selected_customer_id = st.sidebar.selectbox("Select Customer ID", customer_ids)
            
            # Get Customer Data
            cust_data = df[df['CustomerID'] == selected_customer_id].iloc[0]
            
            # Update values
            defaults['Gender'] = cust_data['Gender']
            defaults['Age'] = int(cust_data['Age'])
            defaults['Location'] = cust_data['Location']
            defaults['Tenure'] = int(cust_data['Tenure_Months'])
            defaults['Avg_Spend'] = int(cust_data['Avg_Monthly_Spend'])
            defaults['Last_Spend'] = int(cust_data['Last_Month_Spend'])
            defaults['Transactions'] = int(cust_data['Num_Transactions'])
            defaults['Days_Since'] = int(cust_data['Days_Since_Last_Purchase'])
            defaults['Tickets'] = int(cust_data['Support_Tickets'])
            
            st.info(f"Loaded data for **{selected_customer_id}**")
        else:
            st.error("Customer dataset not found in 'Dataset/customer_data.csv'")

    # --- Input Form ---
    with st.sidebar.form("prediction_form"):
        st.subheader("Customer Details")
        
        # We use the defaults dictionary to set the initial values
        # Note: 'value' in widgets sets the default.
        
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults['Gender']=="Male" else 1)
        age = st.number_input("Age", min_value=18, max_value=100, value=defaults['Age'])
        
        # Map location string to index for selectbox
        loc_options = ["Urban", "Suburban", "Rural"]
        try:
            loc_index = loc_options.index(defaults['Location'])
        except:
            loc_index = 0
            
        location = st.selectbox("Location", loc_options, index=loc_index)
        tenure = st.number_input("Tenure (Months)", min_value=0, value=defaults['Tenure'])
        
        st.divider()
        
        avg_spend = st.number_input("Avg Monthly Spend ($)", min_value=0, value=defaults['Avg_Spend'])
        last_spend = st.number_input("Last Month Spend ($)", min_value=0, value=defaults['Last_Spend'])
        transactions = st.number_input("Num Transactions", min_value=0, value=defaults['Transactions'])
        days_since = st.number_input("Days Since Last Purchase", min_value=0, value=defaults['Days_Since'])
        support_tickets = st.number_input("Support Tickets", min_value=0, value=defaults['Tickets'])
        
        submit = st.form_submit_button("Predict")

    # --- Prediction & Display ---
    
    # Placeholder for main content intro
    if not submit:
        st.markdown("""
        ### Welcome!
        This system predicts customer behavior using advanced Machine Learning models.
        
        **How to use:**
        1. Select **"Existing Customer"** in the sidebar to load real data, OR
        2. Select **"Manual Entry"** to test hypothetical scenarios.
        3. Adjust any values in the form.
        4. Click **Predict** to see the results.
        """)
        
        if input_mode == "Existing Customer" and selected_customer_id:
             st.write(f"Currently viewing data for: **{selected_customer_id}**")
             st.dataframe(df[df['CustomerID'] == selected_customer_id])


    if submit:
        # Prepare data dictionary
        raw_data = {
            'Age': age,
            'Gender': gender,
            'Tenure_Months': tenure,
            'Avg_Monthly_Spend': avg_spend,
            'Last_Month_Spend': last_spend,
            'Num_Transactions': transactions,
            'Days_Since_Last_Purchase': days_since,
            'Support_Tickets': support_tickets,
            'Location': location
        }
        
        try:
            # Preprocess
            processed_data = preprocess_input(raw_data, models)
            
            # Predict
            cluster = models['kmeans'].predict(processed_data)[0]
            churn_risk = models['classification'].predict(processed_data)[0]
            next_spend = models['regression'].predict(processed_data)[0]
            
            # Display Results
            st.divider()
            st.subheader(f"Prediction Results {'for ' + selected_customer_id if selected_customer_id else ''}")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.info("Customer Segment")
                st.metric(label="Cluster", value=f"Group {cluster}")
                
            with c2:
                st.warning("Churn Risk") if churn_risk == 1 else st.success("Churn Risk")
                risk_label = "High Risk" if churn_risk == 1 else "Low Risk"
                st.metric(label="Status", value=risk_label)
                
            with c3:
                st.success("Future Value")
                st.metric(label="Predicted Next Month Spend", value=f"${next_spend:.2f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write(e)

else:
    st.warning("Please ensure all model files (.pkl) are present in the 'pkl' directory.")

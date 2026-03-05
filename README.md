<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a1a,40:1a0d2e,100:7c3aed&height=220&section=header&text=🧠%20Smart%20Sales&fontSize=68&fontColor=ffffff&fontAlignY=40&desc=Customer%20Intelligence%20System%20%7C%20ML%20%2B%20Streamlit&descColor=a78bfa&descAlignY=62&animation=fadeIn" />

<br/>

<img src="https://img.shields.io/badge/FRAMEWORK-Streamlit-FF4B4B?style=for-the-badge&labelColor=0a0a1a&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/ML%20MODELS-3-a78bfa?style=for-the-badge&labelColor=0a0a1a"/>
<img src="https://img.shields.io/badge/FEATURES-9-7c3aed?style=for-the-badge&labelColor=0a0a1a"/>
<img src="https://img.shields.io/badge/CLUSTERS-K%3D3-22d3ee?style=for-the-badge&labelColor=0a0a1a"/>

<br/><br/>

> *Input a customer. Get their segment, churn risk, and next month's spend — instantly.*

</div>

---

## 🎯 What It Does

Three ML models. One dashboard. Actionable intelligence in seconds.

<div align="center">

<table>
  <tr>
    <td align="center" width="240">
      <h3>🔵 Segmentation</h3>
      <b>K-Means Clustering</b><br/>
      <sub>Groups customers into 3 behavioral clusters for targeted marketing strategies</sub>
    </td>
    <td align="center" width="240">
      <h3>🔴 Churn Risk</h3>
      <b>Logistic Regression</b><br/>
      <sub>Predicts High Risk vs Low Risk — act before the customer leaves</sub>
    </td>
    <td align="center" width="240">
      <h3>🟢 Future Value</h3>
      <b>Linear Regression</b><br/>
      <sub>Forecasts exact $ spend next month for revenue planning</sub>
    </td>
  </tr>
</table>

</div>

---

## 🔬 ML Pipeline — Deep Dive

<div align="center">

```
  RAW INPUT (9 Features)
  Age · Gender · Location · Tenure · Avg Monthly Spend
  Last Month Spend · Num Transactions · Days Since Purchase · Support Tickets
          │
          ▼
  ┌───────────────────────┐
  │   DATA PREPROCESSING   │
  │  Label Encode Gender   │
  │  One-Hot Encode Location│
  │  StandardScaler        │
  └───────────────────────┘
          │
          ▼
  ┌───────────────────────┐
  │         PCA            │  → Dimensionality reduction
  │   (pca.pkl loaded)     │    Captures max variance
  └───────────────────────┘
          │
    ┌─────┴──────┬──────────────┐
    ▼            ▼              ▼
┌────────┐  ┌─────────┐  ┌──────────┐
│K-Means │  │Logistic │  │  Linear  │
│Cluster │  │Regress. │  │Regress.  │
│kmeans  │  │Classif. │  │Regress.  │
│.pkl    │  │_Model   │  │_Model    │
│        │  │.pkl     │  │.pkl      │
└────────┘  └─────────┘  └──────────┘
    │            │              │
    ▼            ▼              ▼
 Group 0/1/2  🔴 High Risk   💵 $XXX.XX
              🟢 Low Risk    next month
```

</div>

---

## 🖥️ App Workflow

```
1. OPEN DASHBOARD
   └── Sidebar: Choose input mode
         ├── 📋 Existing Customer  →  Select Customer ID
         │                             Auto-populates all fields from CSV
         └── ✏️  Manual Entry      →  Fill in 9 feature fields manually

2. HIT PREDICT
   └── Inputs encoded → scaled → PCA transformed
         └── Passed simultaneously to all 3 pkl models

3. VIEW RESULTS
   ├── 🔵 Customer Segment    →  Group 0 / 1 / 2
   ├── 🔴🟢 Churn Risk        →  High Risk (red) or Low Risk (green)
   └── 💵 Predicted Spend     →  $XXX.XX next month
```

---

## 🧪 Model Details

<details>
<summary><b>🔴 Churn Classification — Logistic Regression</b></summary>

<br/>

- **Tuned params:** `C`, `penalty`, `solver`, `class_weight`
- **Method:** RandomizedSearchCV — 5-fold Cross-Validation
- **Output:** Binary — `High Risk (1)` or `Low Risk (0)`
- **Metric:** Accuracy, Precision, Recall, F1-score

</details>

<details>
<summary><b>🟢 Spend Regression — Linear Regression</b></summary>

<br/>

- **Tuned params:** `fit_intercept`, `positive`
- **Method:** RandomizedSearchCV to minimize MAE & RMSE
- **Output:** Continuous — predicted $ spend next month
- **Metric:** MAE, RMSE

</details>

<details>
<summary><b>🔵 Customer Clustering — K-Means</b></summary>

<br/>

- **k = 3** — determined by Elbow Method (WCSS vs k plot)
- **Validated** with Silhouette Score
- **Output:** Cluster label — Group 0, 1, or 2
- Trained on PCA-reduced feature space

</details>

---

## ⚙️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python&theme=dark"/>
<br/><br/>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/joblib-4B5563?style=flat-square"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square"/>
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square"/>

</div>

---

## 📁 Project Structure

```
Smart-Sales/
├── app.py                        # Streamlit UI + inference logic
├── Dataset/
│   ├── customer_data.csv         # Raw customer profiles
│   └── preprocessed_data.csv     # Cleaned & encoded training data
├── Models/
│   ├── Data_Preprocessing.py     # Cleaning, encoding, train/test split
│   ├── Classification_Model.py   # Trains churn logistic regression
│   ├── Regression_Model.py       # Trains spend linear regression
│   └── Unsupervised_model.py     # Trains K-Means with Elbow method
└── pkl/
    ├── scaler.pkl                 # StandardScaler artifact
    ├── pca.pkl                    # PCA model artifact
    ├── gender_encoder.pkl         # LabelEncoder for Gender
    ├── Classification_Model.pkl   # Churn prediction model
    ├── Regression_Model.pkl       # Spend forecast model
    └── kmeans_model.pkl           # Customer clustering model
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Ronit178693/Smart-Sales-Custom.git
cd Smart-Sales-Custom

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl files)
python Models/Data_Preprocessing.py
python Models/Classification_Model.py
python Models/Regression_Model.py
python Models/Unsupervised_model.py

# Launch the dashboard
streamlit run app.py
```

---

## 📈 Business Impact

| Intelligence | Business Use |
|---|---|
| 🔵 **Customer Segments** | Tailor campaigns per cluster — stop generic blasting |
| 🔴 **Churn Risk** | Trigger retention offers before the customer leaves |
| 💵 **Future Spend** | Forecast next month's revenue with customer-level precision |
| 📊 **Combined View** | Prioritize high-value, low-churn-risk customers for upsells |

---

<div align="center">

**Built with 🧠 by [Ronit Agrawal](https://github.com/Ronit178693)**

<img src="https://img.shields.io/badge/Open%20to%20Feedback-a78bfa?style=for-the-badge&labelColor=0a0a1a"/>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c3aed,60:1a0d2e,100:0a0a1a&height=120&section=footer&animation=fadeIn" />

</div>

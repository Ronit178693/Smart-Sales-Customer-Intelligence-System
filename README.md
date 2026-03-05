<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0f1a,40:0d2137,100:00d4ff&height=220&section=header&text=🧠%20Smart%20Sales&fontSize=68&fontColor=ffffff&fontAlignY=40&desc=Customer%20Intelligence%20%26%20ML%20Analytics%20Platform&descColor=00d4ff&descAlignY=62&animation=fadeIn" />

<br/>

<img src="https://img.shields.io/badge/LANGUAGE-Python-3776AB?style=for-the-badge&labelColor=0a0f1a&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TYPE-ML%20Pipeline-00d4ff?style=for-the-badge&labelColor=0a0f1a"/>
<img src="https://img.shields.io/badge/MODELS-3+-a78bfa?style=for-the-badge&labelColor=0a0f1a"/>
<img src="https://img.shields.io/badge/NOISE%20REDUCED-40%25-22d3ee?style=for-the-badge&labelColor=0a0f1a"/>

<br/><br/>

> *Transform raw customer data into business intelligence — segment, predict, and retain.*

</div>

---

## 📉 The Business Problem

```
❌  Businesses lose customers without knowing why — until it's too late
❌  Marketing spend wasted on low-value or already-churned customers
❌  No way to distinguish high-value segments from average ones
❌  Raw transaction data is noisy, inconsistent, and hard to act on
```

## ✅ What Smart Sales Does

```
✔  Cleans & normalizes raw data — 40% feature noise reduction via preprocessing
✔  Segments customers into behavioral clusters using K-Means + Elbow/Silhouette
✔  Predicts churn probability before it happens — act, don't react
✔  Identifies high-value customers for targeted retention campaigns
✔  Outputs actionable insights, not just model accuracy numbers
```

---

## 🔬 ML Pipeline — End to End

<div align="center">

```
  RAW DATA
     │
     ▼
┌─────────────────────┐
│   DATA PREPROCESSING │  → Outlier detection, normalization, encoding
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  FEATURE ENGINEERING │  → PCA (95% variance retained), t-SNE, selection
└─────────────────────┘
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
┌──────────────────┐               ┌──────────────────────┐
│  UNSUPERVISED ML  │               │    SUPERVISED ML      │
│  K-Means Cluster  │               │  Logistic Regression  │
│  Elbow + Silhouett│               │  Random Forest        │
│  Customer Segments│               │  Gradient Boosting    │
└──────────────────┘               └──────────────────────┘
     │                                      │
     └──────────────┬───────────────────────┘
                    ▼
         ┌─────────────────────┐
         │   BUSINESS INSIGHTS  │
         │  Churn Probability   │
         │  High-Value Segments │
         │  Sales Opportunities │
         └─────────────────────┘
```

</div>

---

## 🎯 Key Features

<table>
  <tr>
    <td width="50%">

**🔵 Unsupervised Learning**
- K-Means customer segmentation
- Elbow method + Silhouette score validation
- Behavioral cluster profiling

**🟠 Supervised Learning**
- Logistic Regression (baseline)
- Random Forest (ensemble)
- Gradient Boosting (boosted)
- RandomizedSearchCV hyperparameter tuning (5-fold CV)

    </td>
    <td width="50%">

**🟣 Feature Engineering**
- PCA — retains 95% variance, reduces dimensionality
- t-SNE for cluster visualization
- Outlier detection & removal
- Label encoding + feature scaling

**🟢 Model Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Cross-validation for generalization
- Exportable insight reports

    </td>
  </tr>
</table>

---

## 📊 Business Outputs

```
📌  Customer Segments       →  Who are your customers, behaviorally?
📉  Churn Probability       →  Who is about to leave — and when?
💎  High-Value Customers    →  Who drives 80% of your revenue?
📈  Sales Opportunity Score →  Where should you focus next quarter?
```

---

## ⚙️ Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python&theme=dark"/>
<br/><br/>

<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square"/>
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square"/>
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/VS%20Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white"/>

</div>

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Ronit178693/Smart-Sales-Custom.git
cd Smart-Sales-Custom

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

---

## 📁 Project Structure

```
Smart-Sales/
├── data/
│   ├── raw/                  # Original customer dataset
│   └── processed/            # Cleaned & encoded data
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_churn_prediction.ipynb
├── models/
│   └── trained_models/       # Saved model artifacts
├── outputs/
│   └── insights/             # Exported business reports
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── cluster.py
│   └── predict.py
└── requirements.txt
```

---

## 📈 Expected Business Impact

| Metric | Impact |
|--------|--------|
| 📉 Customer Churn | Early detection → proactive retention |
| 💰 Customer LTV | Focus resources on high-value segments |
| 🎯 Marketing ROI | Targeted campaigns, not spray-and-pray |
| 🔍 Sales Efficiency | Score opportunities before pursuing them |

---

<div align="center">

**Built with 🧠 by [Ronit Agrawal](https://github.com/Ronit178693)**

<img src="https://img.shields.io/badge/Open%20to%20Feedback-00d4ff?style=for-the-badge&labelColor=0a0f1a"/>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,60:0d2137,100:0a0f1a&height=120&section=footer&animation=fadeIn" />

</div>

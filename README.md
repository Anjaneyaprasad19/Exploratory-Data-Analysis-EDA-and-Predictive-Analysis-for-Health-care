# Healthcare Predictive Analysis

A Streamlit-based web application for exploratory data analysis (EDA), predictive modeling, and personalized recommendations in healthcare. This project leverages machine learning to predict medical test results and provide actionable insights for patients and practitioners.

---

## 🚀 Features

- **User Authentication**: Simple login system for secure access.
- **Data Exploration**: Interactive EDA and feature engineering visualizations.
- **Model Building & Evaluation**: Train and evaluate multiple ML models (Random Forest, SVM, Logistic Regression, etc.).
- **Algorithm Predictions**: Run and compare different algorithms on cleaned healthcare data.
- **Future Predictions**: Predict patient test results and get personalized health recommendations.
- **Lifestyle Recommendations**: Condition- and medication-based lifestyle tips.
- **Contact & Info**: Project details and contact information.

---

## 🗂️ Project Structure

```
healthcare-predictive-analysis/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── analysis_pages/         # HTML files for EDA and modeling sections
│   ├── Data Loading and Preprocessing.html
│   ├── EDA.html
│   ├── Feature Engineering.html
│   └── Model Building and Evaluation.html
│
├── data/                   # Datasets (not tracked by git)
│   ├── Cleaned_healthcare_dataset.csv
│   └── Project_healthcare_dataset.csv
│
├── models/                 # Trained model files (not tracked by git)
│   ├── label_encoders.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/              # Jupyter notebooks for EDA and modeling
│   ├── Cross.ipynb
│   ├── Data Loading and Preprocessing.ipynb
│   ├── Demo Random.ipynb
│   ├── EDA.ipynb
│   ├── Feature Engineering.ipynb
│   ├── Feature Importance.ipynb
│   ├── Model Building and Evaluation.ipynb
│   ├── ROC.ipynb
│   ├── SMOTE Model Building.ipynb
│   └── Testing.ipynb
```

---

## ⚙️ Setup & Usage

1. **Clone the repository**
   ```powershell
   git clone https://github.com/Anjaneyaprasad19/Exploratory-Data-Analysis-EDA-and-Predictive-Analysis-for-Health-care.git
   cd Exploratory-Data-Analysis-EDA-and-Predictive-Analysis-for-Health-care
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Add your data and models**
   - Place `Project_healthcare_dataset.csv` and `Cleaned_healthcare_dataset.csv` in the `data/` folder.
   - Place `random_forest_model.pkl`, `scaler.pkl`, and `label_encoders.pkl` in the `models/` folder.

4. **Run the Streamlit app**
   ```powershell
   streamlit run app.py
   ```

---

## 📊 Notebooks
- All EDA, feature engineering, and model development steps are documented in the `notebooks/` folder for reproducibility and further exploration.

---

## 📁 Data & Models
- **data/** and **models/** are excluded from git tracking (see `.gitignore`).
- Use sample or synthetic data for demo purposes. For real data, ensure compliance with privacy regulations.

---

## 📝 License
This project is for educational and demonstration purposes. For commercial or clinical use, further validation and compliance are required.

---

## 📬 Contact
- **Email:** healthcare@example.com
- **Address:** 

---

**Data Privacy Notice:** The data used is synthetic. For real data, ensure compliance with data protection regulations.

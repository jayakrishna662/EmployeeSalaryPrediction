# 💼 Employee Salary Prediction Using XGBoost Classifier

This project aims to predict employee salaries whether it is more than 50k or less than/equal to 50k using machine learning algorithm **XGBoost** based on various personal and work-related features.  
The application is built using Python leveraging libraries like **pandas**, **numpy**, **scikit-learn**, **matplotlib**, **seaborn**, **ngrok**, and **streamlit** for interactive web application.

---

## 📌 Overview

The Employee Salary Prediction application takes user input of various features through the **Streamlit** interface and predicts the employee's salary when clicked on the **Predict** button using the pre-trained ML model.

The model is trained on the **UCI Adult Income dataset**, a popular dataset used for income classification tasks.

---

## 📊 Features Used

- Age  
- Workclass  
- Final Weight (fnlwgt)  
- Education Number  
- Marital Status  
- Occupation  
- Relationship  
- Race  
- Gender  
- Capital Gain  
- Capital Loss  
- Hours Per Week  
- Native Country  

---

## 📁 Project Files
```text
EmployeeSalaryPrediction/
│
├── app.py                          # Streamlit web app
├── run_with_ngrok.py              # Script to expose the app online using ngrok
├── employeeSalaryPrediction.ipynb # Jupyter Notebook (full ML pipeline)
├── xgb_model.pkl                  # Trained model
├── encoders.pkl                   # Encoders for categorical features
├── X_test.pkl                     # Test data (features)
├── y_test.pkl                     # Test data (labels)
├── adult 3.csv                    # Original dataset
└── README.md                      # Project overview

```
---

## 🔧 Steps in Building the Project

### Step 1: Data Collection  
The dataset (UCI Adult Income dataset) is loaded using pandas. It contains demographic and employment information such as age, education, occupation, etc., along with the target variable: income (>50K or <=50K).

### Step 2: Data Preprocessing  
- Handle missing or inconsistent values.  
- Convert categorical variables (e.g., gender, occupation) into numerical form using LabelEncoder from scikit-learn.  
- Normalize or scale numerical features if necessary.  
- Split the dataset into training and testing sets.

### Step 3: Model Selection & Training  
- Use XGBoost Classifier (XGBClassifier) to train a binary classification model.  
- Fit the model on the training set (X_train, y_train).

### Step 4: Model Evaluation  
- Evaluate the model performance using metrics like **accuracy**, **recall**, **precision**, **confusion matrix**, **ROC-AUC**, etc.

### Step 5: Model Performance Improvement  
- **Class Imbalance Analysis**  
  Observed that the dataset had an imbalance between income classes (<=50K vs >50K). Computed the imbalance ratio and used it to adjust the `scale_pos_weight` parameter in XGBoost.  
- **Retraining with scale_pos_weight**  
  Trained a new XGBoost model with the adjusted weight to better learn from the minority class (>50K).  
- **Threshold Tuning**  
  Calculated precision-recall curves and F1-scores for different thresholds.  
  Selected the optimal probability threshold that maximized F1-score.  
  Applied this threshold to convert predicted probabilities into class labels for final evaluation.

### Step 6: Model Serialization  
Saved the trained model, encoders, and test data using `pickle` for use in deployment.

### Step 7: Streamlit App Development  
Created an interactive web application using Streamlit.  
The app allows users to input feature values and get a prediction of income class in real time.

### Step 8: Deployment using Ngrok  
Used **ngrok** to generate a public URL for easy hosting so that we can share and access the application online.

---
## Output
<img width="669" height="814" alt="Screenshot 2025-07-21 204557" src="https://github.com/user-attachments/assets/4696ae64-d6f6-45d5-9546-c876020cc9c5" />

## ▶️ Running the Application

### 🖥️ To Run Locally:
```bash
streamlit run app.py
```
### 🌐 To Run Online with Ngrok:
```bash
python run_with_ngrok.py
```

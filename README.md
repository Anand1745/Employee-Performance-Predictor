# 📊 Employee Performance Predictor (ML + Streamlit)

## 🚀 Overview

This project predicts employee performance using Machine Learning and provides an interactive dashboard for HR analytics.

It helps simulate how factors like experience, salary, and training impact employee performance.

---

## 🎯 Problem Statement

Organizations often struggle to:

* Identify high-performing employees
* Detect low performers early
* Optimize training and salary strategies

This project solves the problem using **data-driven decision making**.

---

## 💡 Solution

A Machine Learning system that:

* Predicts employee performance (High / Medium / Low)
* Provides interactive analysis via Streamlit dashboard
* Enables "What-if" simulation for HR decision making

---

## 🧠 Features

* 📌 Synthetic HR dataset generation
* 🤖 Machine Learning model (Random Forest)
* 📊 Interactive Streamlit dashboard
* 🎯 Real-time performance prediction
* 📈 Feature importance visualization
* 🔮 Performance simulation (What-if analysis)
* 📉 Insights & analytics

---

## 🏗️ Project Architecture

```
Data → Preprocessing → Model Training → Evaluation → Streamlit Dashboard → Insights
```

---

## 📁 Folder Structure

```
Employee-Performance-Predictor/
│
├── data/              # Dataset
├── src/               # Core ML pipeline
│   ├── data_gen.py
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
│
├── models/            # Saved model (.pkl)
├── images/            # Screenshots
├── outputs/           # Graph outputs
│
├── app.py             # Streamlit dashboard
├── main.py            # Training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/Employee-Performance-Predictor-ML.git
cd Employee-Performance-Predictor-ML
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Generate Data + Train Model

```
python main.py
```

### 4️⃣ Run Dashboard

```
streamlit run app.py
```

---

## 📊 Dashboard Features

* KPI Metrics (Total Employees, Avg Salary, High Performers %)
* Employee Performance Prediction
* Confidence Score
* Dataset Preview
* Insights & Visualization
* Feature Importance
* Simulation (Experience Impact)
* What-if Analysis

---

## 📈 Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: ~85–90%

---

## 🧠 Key Insights

* Training hours strongly influence performance
* Salary has a positive correlation with performance
* Experience contributes but is less dominant

---

## 🎯 Business Value

* Helps HR teams make data-driven decisions
* Improves employee productivity
* Supports promotion and training planning
* Reduces performance-related risks

---

## 🔮 Future Improvements

* Use real HR datasets
* Add employee attrition prediction
* Deploy as a web app
* Add authentication system
* Integrate with databases

---

## 👨‍💻 Author

**Anand Ramesh Karunakaran**

---

## 🌟 If you like this project

Give it a ⭐ on GitHub!

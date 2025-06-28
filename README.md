# 🚢 Titanic Survival - In-Depth Analysis

## 🧩 Problem Statement

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On **April 15, 1912**, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Sadly, there weren’t enough lifeboats for everyone onboard — resulting in the death of **1502 out of 2224** passengers and crew.

While some survivors owed their fate to luck, certain **groups of people were statistically more likely to survive** than others.

---

## 🎯 Project Objective

Build a **predictive model** that answers the key question:

> "What kinds of people were more likely to survive?"

Using real Titanic passenger data — such as **name, age, gender, and socio-economic class** — this project explores, analyzes, and builds machine learning models to make survival predictions.

---

## 💻 Project Overview

This project was developed as part of the Kaggle challenge:  


Key components include:

- 🧹 **Data Cleaning & Preprocessing**  
  Handled missing values, encoded categorical features, and engineered new features (e.g., `IsAlone`, `AgeGroup`, `FareBin`, `Title_encoded`).

- 📊 **Exploratory Data Analysis (EDA)**  
  Explored survival trends by age, gender, fare, class, and embarked location using **Matplotlib** and **Seaborn** (bar plots, violin plots, KDEs, boxplots).

- 🤖 **Machine Learning Models**  
  Implemented and compared various classification models:
  - Logistic Regression
  - Support Vector Machines
  - Decision Tree & Random Forest
  - K-Nearest Neighbors
  - Gradient Boosting (Best Performer)
  - Gaussian NB, LDA, QDA

- 🔧 **Model Optimization**  
  Tuned hyperparameters using `GridSearchCV` for better accuracy.

- 📤 **Kaggle Submission**  
  Final predictions were generated and submitted — achieving a score of **0.80125** on the public leaderboard.

---

> This project showcases a complete ML pipeline — from raw data to insight, modeling, evaluation, and deployment.

---


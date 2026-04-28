# Used_cars_deploy-predictor 
# 🚗 Used Car Price Predictor
<img width="474" height="474" alt="image" src="https://github.com/user-attachments/assets/e8a7caa6-ea8a-4411-8b9f-0ccdaab4760c" />

## 📌 Overview
This project predicts the price of used cars based on their characteristics. It combines **Exploratory Data Analysis (EDA)** and **Machine Learning models** to uncover the most influential factors affecting car prices.

## 🎯 Objectives
- Clean and preprocess raw car datasets.
- Perform **EDA** to identify trends and correlations.
- Highlight the **main characteristics impacting car prices** (mileage, brand, year, fuel type, etc.).
- Build and evaluate predictive models.
- Provide clear visualizations and documentation.

## 📂 Repository Structure
- `Exploratory_data_analysis_cars.ipynb` → Notebook for EDA and visualization.
- `data/` → Dataset(s) used for training and analysis.
- `models/` → Trained models and prediction scripts.
- `README.md` → Project documentation.

## 🔑 Key Features
- **Data Cleaning**: Handle missing values, outliers, and categorical encoding.
- **Feature Engineering**: Create attributes like car age, mileage buckets, and brand categories.
- **Visualization**: Heatmaps, scatterplots, and boxplots to show relationships.
- **Modeling**: Regression and tree-based models for price prediction.
- **Evaluation**: Metrics such as RMSE, MAE, and R².
- **deployment**: fastapi , graido 

## 🛠️ Tech Stack
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn)
- Jupyter Notebook
- GitHub for version control

## 📊 Insights
From the analysis, the following characteristics strongly influence car price:
- **Mileage** → Higher mileage lowers price.
- **Car Age** → Newer cars retain higher value.
- **Brand** → Premium brands command higher prices.
- **Fuel Type & Transmission** → Certain combinations affect pricing.
- **Engine Size & Horsepower** → Larger engines often increase value.

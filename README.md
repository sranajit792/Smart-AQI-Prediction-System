🌍 Smart AQI Prediction & Visualization System
📌 Project Overview

The Smart AQI Prediction & Visualization System is a Machine Learning-based application developed to predict Air Quality Index (AQI) using pollutant concentration levels. The system analyzes historical air pollution data and provides accurate AQI predictions along with interactive visualizations and health recommendations.

This project was developed as part of an AICTE-affiliated initiative in collaboration with Microsoft Elevate, focusing on practical implementation of Artificial Intelligence and real-world environmental monitoring solutions.

🎯 Objective

The main objective of this project is to:

Predict AQI using Machine Learning techniques

Provide an interactive dashboard for visualization

Display AQI categories with health advice

Improve awareness regarding air pollution

📊 Dataset

The model is trained using the Air Quality Data in India dataset from Kaggle.

Dataset Source:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Features Used:

PM2.5

PM10

NO₂

CO

SO₂

O₃

AQI (Target Variable)

🤖 Machine Learning Model

The project uses Random Forest Regressor, an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Model Training Steps:

Removed missing AQI values

Filled missing feature values using mean imputation

Split dataset into features and target variable

Trained Random Forest model

Saved trained model as model.pkl

💻 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Plotly

Joblib

ReportLab

📈 Features of the Application

AQI prediction based on pollutant input

AQI category classification (Good, Moderate, Poor, etc.)

Health recommendations based on AQI level

Interactive speedometer gauge

Historical AQI trend visualization

Pollutant radar chart

Live location map visualization

Model performance metrics (R² Score & MAE)

Downloadable AQI PDF report

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/Smart-AQI-Prediction-System.git


Install required libraries:

pip install -r requirements.txt


Train the model:

python train_model.py


Run the Streamlit app:

streamlit run app.py

📌 Project Structure
├── app.py
├── train_model.py
├── data.csv
├── model.pkl
├── requirements.txt
└── README.md

📊 Model Performance

The model performance is evaluated using:

R² Score

Mean Absolute Error (MAE)

The results demonstrate reliable and accurate AQI prediction capability.

🔮 Future Enhancements

Integration of real-time IoT air quality sensors

Implementation of Deep Learning models (LSTM, ANN)

Integration of weather data for improved accuracy

Mobile application development

Cloud deployment using Microsoft Azure

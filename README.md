# Disaster_Management

Machine Learning Based Assessment of Disaster Response Efficiency and Regional Recovery Dynamics

📌 Project Overview

Machine Learning Based Assessment of Disaster Response Efficiency and Regional Recovery Dynamics is a machine-learning project designed to analyze disaster-response data and evaluate how effectively regions respond to disasters and how long they take to recover.

The system performs two related prediction tasks:

Classification: Categorizes disaster response efficiency into three levels:

Low

Medium

High

Regression: Predicts the expected number of recovery days after a disaster.

The project provides a graphical desktop interface built with Tkinter for dataset loading, preprocessing, exploratory data analysis, train/test splitting, model training, model comparison, and prediction.

It also includes a separate Gradio web interface that allows a user to upload a CSV file and receive response-efficiency and recovery-time predictions using trained CatBoost models.

🎯 Objectives

The main objectives of this project are:

Analyze historical disaster-response data.

Study relationships between disaster severity, casualties, economic loss, response time, aid, geography, and recovery.

Preprocess categorical, date, and numerical features.

Convert continuous response-efficiency scores into meaningful Low/Medium/High classes.

Train and compare multiple machine-learning algorithms.

Evaluate classification performance using Accuracy, Precision, Recall, F1-score, and AUC where applicable.

Evaluate regression performance using R², MAE, and RMSE.

Visualize model performance and regression predictions.

Save trained models and preprocessing objects for reuse.

Provide row-wise predictions for new disaster records.

Provide a simple Gradio-based prediction interface.

🧠 Problem Statement

Disaster management requires rapid decisions based on limited and often complex information. Factors such as disaster severity, casualties, economic loss, response time, aid received, and geographical location can influence the effectiveness of emergency response and the time required for regional recovery.

Traditional analysis may make it difficult to identify patterns across multiple factors simultaneously.

This project applies machine learning to historical disaster data to answer two practical questions:

How effective is the disaster response likely to be?

and

How many days may be required for recovery?

The system can therefore be used as an analytical and predictive prototype for disaster-response assessment.

✨ Key Features

1. Dataset Upload

The Tkinter application allows the user to select and load a CSV dataset.

2. Data Preprocessing

The preprocessing pipeline:

Converts the date column into a datetime format.

Extracts:

year

month

Removes the original date column.

Converts country and disaster_type using LabelEncoder.

Standardizes numerical input features using StandardScaler.

Creates three response-efficiency categories using quantile-based binning:

Low

Medium

High

The fitted scaler and label encoders are saved with Joblib for later prediction.

3. Exploratory Data Analysis

The project includes visual analysis such as:

Top 10 most impacted countries.

Casualty distribution.

Classification comparison plots.

Regression comparison plots.

Actual-vs-predicted regression plots.

Confusion matrices.

4. Train/Test Split

The dataset is divided into training and testing sets using:

Test size: 20%

Random state: 42

5. Multiple Machine-Learning Models

The project trains models for both classification and regression.

Implemented model families include:

Decision Tree

AdaBoost

SGD

CatBoost

Additional Random Forest and Extra Trees related code is also present in the project files.

6. Classification

The classification target is:

response_efficiency_score

The original continuous response-efficiency score is divided into three quantile-based classes:

Low
Medium
High

7. Regression

The regression target is:

recovery_days

The model predicts the expected number of days required for recovery.

8. Model Evaluation

Classification Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC where it can be calculated

Regression Metrics

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

9. Model Persistence

Trained models are saved as .pkl files using Joblib.

Preprocessing objects are also stored so that the same transformations can be applied to new data.

10. Prediction System

A new CSV file can be uploaded to generate:

Predicted response efficiency

Predicted recovery days

The Tkinter application displays row-wise prediction results.

11. Gradio Web Interface

The gradio folder contains a separate web-based prediction interface.

The Gradio application:

Accepts a CSV file.

Applies the saved preprocessing pipeline.

Loads the trained CatBoost classifier and regressor.

Generates predictions.

Displays the prediction results as a dataframe.

📊 Dataset

The main dataset is:

Dataset/Disaster Emergency Response.csv

A test dataset is also provided:

Dataset/TestData.csv

Dataset Features

The dataset contains the following fields:

Feature

Description

date

Date of the disaster

country

Country where the disaster occurred

disaster_type

Type/category of disaster

severity_index

Numerical measure of disaster severity

casualties

Number of casualties

economic_loss_usd

Estimated economic loss in USD

response_time_hours

Time taken to respond to the disaster

aid_amount_usd

Financial aid amount in USD

response_efficiency_score

Continuous score representing response efficiency

recovery_days

Number of days required for recovery

latitude

Geographic latitude

longitude

Geographic longitude

🔄 Machine-Learning Workflow

The overall workflow is:

                 ┌──────────────────────┐
                 │   Disaster Dataset   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Data Preprocessing │
                 │                      │
                 │ • Date conversion    │
                 │ • Year / Month       │
                 │ • Label Encoding     │
                 │ • Standard Scaling   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Preparation  │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 ▼                      ▼
       ┌───────────────────┐  ┌────────────────────┐
       │  Classification   │  │     Regression     │
       │                   │  │                    │
       │ Response          │  │ Recovery Days      │
       │ Efficiency        │  │                    │
       │ Low/Medium/High   │  │ Continuous Value   │
       └─────────┬─────────┘  └──────────┬─────────┘
                 │                       │
                 └──────────┬────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Model Training       │
                 │                      │
                 │ • Decision Tree      │
                 │ • AdaBoost           │
                 │ • SGD                │
                 │ • CatBoost           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Prediction / Output  │
                 └──────────────────────┘

🤖 Algorithms

1. Decision Tree

Decision Tree models are used for both:

Classification

Regression

The project uses a maximum tree depth of 5 in the main implementation to control model complexity.

Classification

Predicts:

Low / Medium / High

Regression

Predicts:

Recovery Days

2. AdaBoost

AdaBoost combines multiple weak learners to create a stronger predictive model.

The project uses AdaBoost for:

Response-efficiency classification

Recovery-days regression

3. SGD

Stochastic Gradient Descent based models are included for:

Classification

Regression

They provide a computationally efficient linear-model baseline.

4. CatBoost

CatBoost is used for:

Classification

Regression

The trained CatBoost models are also used by the Gradio prediction application.

CatBoost is particularly useful for structured/tabular datasets and provides a strong model for comparison against the other approaches.

📈 Evaluation Methodology

Classification

The following metrics are calculated:

Accuracy

Measures the proportion of correct predictions.

Accuracy = Correct Predictions / Total Predictions

Precision

Measures how many predicted positive/class assignments are correct.

Recall

Measures how many actual class instances are successfully identified.

F1-score

Combines precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

ROC-AUC

Where probability outputs and class conditions allow it, multiclass ROC-AUC is calculated using a one-vs-rest approach.

Regression

R² Score

Measures how well the model explains the variance in the target.

Higher values generally indicate better explanatory performance.

MAE

Mean Absolute Error measures the average absolute prediction error.

RMSE

Root Mean Squared Error penalizes larger errors more strongly.

🖥️ Tkinter Desktop Application

The main application is implemented in:

main.py

The application window provides buttons for the major stages of the machine-learning workflow.

Available operations

Dataset
Preprocessing
EDA
Train Test Splitting
Decision Tree Model
AdaBoost Model
SGD Model
CatBoost Model
Classification Comparison
Regression Comparison
Prediction
Exit

Recommended execution order

Run the operations in this order:

1. Dataset
2. Preprocessing
3. EDA
4. Train Test Splitting
5. Select/train models
6. Classification Comparison
7. Regression Comparison
8. Prediction

🌐 Gradio Web Application

The web application is located at:

gradio/app.py

It provides a simpler prediction workflow.

Gradio workflow

Upload CSV
     ↓
Read input data
     ↓
Date preprocessing
     ↓
Label encoding
     ↓
Feature scaling
     ↓
CatBoost classification
     ↓
CatBoost regression
     ↓
Display prediction dataframe

The Gradio interface produces two additional columns:

Pred_Response_Efficiency
Pred_Recovery_Days

📁 Project Structure

MINI PROJECT/
│
├── Dataset/
│   ├── Disaster Emergency Response.csv
│   └── TestData.csv
│
├── gradio/
│   ├── app.py
│   ├── requirements.txt
│   ├── catboost_classifier.pkl
│   ├── catboost_regssion.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
├── model/
│   ├── ETC_model.pkl
│   └── RFC_model.pkl
│
├── models/
│   ├── adaboost_classifier.pkl
│   ├── adaboost_regressor.pkl
│   ├── catboost_classifier.pkl
│   ├── catboost_regssion.pkl
│   ├── decision_tree_classifier.pkl
│   ├── decision_tree_regressor.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── sgd_classifier.pkl
│   └── sgd_regressor.pkl
│
├── results/
│   ├── adaboost_actual_vs_pred.png
│   ├── catboost_actual_vs_pred.png
│   ├── decision_tree_actual_vs_pred.png
│   └── sgd_actual_vs_pred.png
│
├── main.py
├── main1.py
├── A18.ipynb
├── response.jpg
└── run.bat

The ZIP also contains Jupyter checkpoint files and CatBoost training/logging artifacts. These are development artifacts and are not required for understanding the core workflow.

⚙️ Requirements

The project uses Python and the following major libraries:

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

CatBoost

Pillow

Tkinter

Gradio

The Gradio-specific dependencies are listed in:

gradio/requirements.txt

Current Gradio requirements include:

pandas
numpy
scikit-learn
matplotlib
seaborn
gradio
joblib
catboost

For the full Tkinter application, Pillow is also required.

🚀 Installation

Step 1: Clone the repository

git clone https://github.com/abhinandu19/Disaster_Management.git
cd Disaster_Management

Step 2: Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

Step 3: Install dependencies

For the Gradio application:

pip install -r gradio/requirements.txt

For the Tkinter desktop application, also install Pillow:

pip install pillow

If required:

pip install pandas numpy scikit-learn matplotlib seaborn joblib catboost

▶️ Running the Tkinter Application

From the project root:

python main.py

The application will open a desktop GUI.

Important

The application expects response.jpg to be available in the same working directory because it is used as the GUI background image.

▶️ Running the Gradio Application

Move into the Gradio directory:

cd gradio

Install dependencies:

pip install -r requirements.txt

Run:

python app.py

Gradio will start a local web interface and display the local URL in the terminal.

📥 Input Format for Prediction

A prediction CSV should contain the feature columns expected by the trained preprocessing pipeline.

Typical input columns are:

date
country
disaster_type
severity_index
casualties
economic_loss_usd
response_time_hours
aid_amount_usd
latitude
longitude

For prediction, the target columns should generally not be included:

response_efficiency_score
recovery_days

The application derives:

year
month

from the date column.

📤 Prediction Output

The system adds:

Pred_Response_Efficiency
Pred_Recovery_Days

Example conceptual output:

country     disaster_type    Pred_Response_Efficiency    Pred_Recovery_Days
India       Earthquake       High                         42.7
Brazil      Hurricane        Medium                       58.3

The exact prediction values depend on the trained models and uploaded input data.

💾 Saved Models

The project contains serialized models in .pkl format.

Examples include:

models/decision_tree_classifier.pkl
models/decision_tree_regressor.pkl
models/adaboost_classifier.pkl
models/adaboost_regressor.pkl
models/sgd_classifier.pkl
models/sgd_regressor.pkl
models/catboost_classifier.pkl
models/catboost_regssion.pkl

Preprocessing objects:

models/scaler.pkl
models/label_encoders.pkl

These files allow trained models and preprocessing transformations to be reused without retraining every time.

📊 Results and Visualizations

The results/ directory contains generated regression comparison visualizations such as:

adaboost_actual_vs_pred.png
catboost_actual_vs_pred.png
decision_tree_actual_vs_pred.png
sgd_actual_vs_pred.png

The application can also generate:

Confusion matrices

Classification comparison charts

Regression comparison charts

Actual-vs-predicted plots

These visualizations help compare different algorithms.

🔬 Data Preprocessing Details

Date Processing

The original date is converted using:

pd.to_datetime(...)

Then:

year = date.year
month = date.month

The original date field is removed after feature extraction.

Categorical Encoding

The following categorical columns are encoded:

country
disaster_type

LabelEncoder is used.

Numerical Scaling

Numerical features are standardized with:

StandardScaler

This transforms numerical features to a standardized scale based on the training data.

Classification Target Creation

The original continuous:

response_efficiency_score

is converted into three quantile-based categories:

Low
Medium
High

using pandas.qcut.

This makes it possible to perform multiclass classification while retaining the original continuous score for analysis before categorization.

🔐 Reproducibility

The main training workflow uses:

random_state = 42

and a:

20% test split

This helps make the train/test split reproducible.

For production use, additional practices such as cross-validation, experiment tracking, model versioning, and independent validation datasets are recommended.

⚠️ Important Project Limitations

This project is an academic/prototype machine-learning system and should not be treated as a production disaster-management decision system without further validation.

Important limitations include:

The dataset is historical and may not represent every real-world disaster situation.

Predictions depend strongly on the quality and distribution of the training data.

Quantile-based Low/Medium/High classes are relative to the dataset distribution.

Label encoding introduces numerical representations for categorical values.

A model may encounter categories during prediction that were not present during training.

The current Gradio preprocessing contains a fallback for unknown object columns; production systems should handle unseen categories more rigorously.

Model performance should be validated on independent real-world datasets.

Disaster-response decisions should not rely solely on machine-learning predictions.

The project does not provide a guarantee of real-world recovery time or response effectiveness.

🛠️ Recommended Improvements

Future versions can improve the system by adding:

Cross-validation.

Hyperparameter tuning.

Feature importance and SHAP explanations.

More advanced ensemble models.

XGBoost/LightGBM comparisons.

Time-series forecasting.

Geographic/geospatial analysis.

Interactive maps.

Real-time disaster data integration.

Real-time weather and satellite data.

Automated data pipelines.

Model monitoring.

REST API deployment.

Cloud deployment.

User authentication.

Database integration.

Docker deployment.

Better handling of unseen categorical values.

Automated model retraining.

Explainable AI dashboards.

Disaster-specific models.

Early-warning and risk scoring capabilities.

🧪 Example Use Case

Suppose a new disaster record contains:

Country: India
Disaster Type: Earthquake
Severity Index: 7.2
Casualties: 150
Economic Loss: 10000000
Response Time: 8 hours
Aid Amount: 500000
Latitude: 17.4
Longitude: 78.5

The system can process the record using the saved preprocessing objects and provide:

Response Efficiency: Predicted class
Recovery Days: Predicted numerical value

This demonstrates how historical disaster information can be transformed into predictive insights.

📚 Technologies Used

Technology

Purpose

Python

Core programming language

Pandas

Data loading and manipulation

NumPy

Numerical computation

Scikit-learn

Machine-learning models and metrics

CatBoost

Classification and regression

Matplotlib

Data visualization

Seaborn

Statistical visualization

Joblib

Model serialization

Tkinter

Desktop GUI

Gradio

Web-based prediction interface

Pillow

Image handling for GUI

Jupyter Notebook

Experimentation and analysis

🧩 Main Files

main.py

Primary Tkinter application containing:

Dataset upload

Preprocessing

EDA

Train/test splitting

Model training

Evaluation

Visualization

Prediction

main1.py

An additional Tkinter-based implementation containing similar preprocessing, EDA, model-training, and evaluation functionality.

gradio/app.py

Web-based prediction application using saved CatBoost models.

A18.ipynb

Jupyter Notebook used for project experimentation/analysis.

Dataset/

Contains the main disaster-response data and test data.

models/

Contains trained machine-learning models and preprocessing artifacts.

results/

Contains generated visualization outputs.

👥 Project Contribution Areas

A typical project team can divide responsibilities into:

Dataset collection and preparation

Exploratory data analysis

Data preprocessing

Machine-learning model development

Model evaluation

GUI development

Gradio/web-interface development

Testing and validation

Documentation and presentation

📌 Academic Project Information

Project Title:

Machine Learning Based Assessment of Disaster Response Efficiency and Regional Recovery Dynamics

Project Domain:

Machine Learning / Data Science / Disaster Management

Primary Tasks:

Classification

Regression

Exploratory Data Analysis

Predictive Analytics

Interfaces:

Tkinter Desktop GUI

Gradio Web Interface

🤝 Contributing

Contributions and improvements are welcome.

A suggested contribution workflow is:

git clone https://github.com/abhinandu19/Disaster_Management.git
cd Disaster_Management

Create a new branch:

git checkout -b feature/your-feature-name

Make your changes, commit them:

git add .
git commit -m "Add your feature"

Push the branch:

git push origin feature/your-feature-name

Then open a Pull Request on GitHub.

📄 License

This project is intended primarily for academic and educational purposes.

If this repository is later distributed as an open-source project, a specific license such as MIT can be added after confirming the project's ownership and licensing requirements.

👤 Author

Abhinandu Mustyala

GitHub:

https://github.com/abhinandu19

Project Repository:

https://github.com/abhinandu19/Disaster_Management

⭐ Acknowledgement

This project demonstrates how machine-learning techniques can be applied to structured disaster-response data to support analytical assessment and predictive modeling.

The system is intended as an academic prototype for learning and experimentation in:

Machine Learning

Data Science

Predictive Analytics

Disaster Management

Human-Computer Interaction

Model Evaluation

⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and sharing suggestions for improving the project.

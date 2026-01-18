# Heart Disease Prediction using Logistic Regression

## Project Overview
This project implements a **logistic regression model** to predict the risk of heart disease (heart attack) based on patient medical data. The work was completed as part of the *Machine Learning* course project and follows a complete supervised learning workflow, from data exploration to model evaluation and interpretation.

The primary goal is to demonstrate a clear, well-documented application of logistic regression for a **binary medical classification problem**, with emphasis on interpretability and proper evaluation metrics.



## Dataset
The dataset used in this project is the **Heart Attack Prediction Dataset** from Kaggle:

- Source: https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset
- Target variable: Heart attack risk (0 = no risk, 1 = risk)
- Features include:
  1.	Age 
  2.	Sex
  3.	Cholesterol levels
  4.	Blood pressure
  5.	Heart Rate
  6.	Diabetics
  7.	Family History
  8.	Smoking
  9.	Obesity
  

The dataset is publicly available and was used exclusively for academic purposes.



## Methodology
The project follows these main steps:

1. **Dataset Exploration**  
   Analysis of feature distributions, target variable balance, and correlations between variables.

2. **Data Preprocessing**  
   - Encoding of categorical variables  
   - Feature scaling using standardization  
   - Preparation of the final feature matrix

3. **Train–Test Split**  
   Stratified splitting to preserve class distribution and ensure reproducibility.

4. **Logistic Regression Model**  
   Implementation of a logistic regression classifier using solver-based optimization and regularization.

5. **Model Training**  
   Training the model on the training dataset and monitoring convergence.

6. **Prediction and Evaluation**  
   Model evaluation using multiple metrics, including accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and precision–recall curve.

7. **Results Interpretation**  
   Analysis of model coefficients to assess feature importance and interpret medical relevance.



## Evaluation Metrics
The following metrics are used to assess model performance:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Confusion Matrix
- ROC Curve (optional)
- Precision–Recall Curve (optional)

These metrics provide a comprehensive evaluation, particularly important for medical prediction tasks.




## How to Run the Code

1. Clone the repository:
   
   git clone https://github.com/USERNAME/heart-disease-logistic-regression.git
   

2. Open the Jupyter notebook:
   
   jupyter notebook mlheart.ipynb
   

3. Run all cells to reproduce the results.



## Requirements
The project uses standard Python data science libraries:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

All dependencies can be installed using pip if not already available.



## Academic Context
This repository is part of a university machine learning project. The implementation and documentation were created for educational purposes and are not intended for clinical use.



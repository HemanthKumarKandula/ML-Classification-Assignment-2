# Online Shoppers Purchasing Intention – ML Classification

## 1\. Problem Statement

The objective of this project is to predict whether an online shopping session results in a purchase using machine learning classification algorithms.

Five classification models are implemented and evaluated to compare their performance.

## 2\. Dataset Description

The Online Shoppers Purchasing Intention dataset is obtained from the UCI Machine Learning Repository.

* Original records: 12,330
* Records after cleaning: 12,205
* Input features: 17
* Target variable: Revenue
* Problem type: Binary classification

The dataset contains information about visitor browsing activity, page visits, time spent on pages and other shopping session details.

Dataset source:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## 3\. GitHub Repository

https://github.com/HemanthKumarKandula/ML-Classification-Assignment-2

## 4\. Data Preprocessing

* Removed 125 duplicate records.
* Separated input features and target variable.
* Used an 80:20 stratified train-test split.
* Applied StandardScaler to numerical features.
* Applied OneHotEncoder to categorical features.
* Used Scikit-learn pipelines to avoid data leakage.

## 5\. Models and Evaluation Results

The five classification algorithms were evaluated using the same test dataset containing 2,441 records.

|Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.8460|0.9097|0.5050|0.7958|0.6179|0.5488|
|Decision Tree|0.8472|0.8449|0.5081|0.7435|0.6036|0.5274|
|KNN|0.8718|0.8003|0.6533|0.3848|0.4843|0.4357|
|Naive Bayes|0.2773|0.7502|0.1756|0.9791|0.2978|0.1375|
|Support Vector Machine|0.8689|0.9095|0.5589|0.7696|0.6476|0.5805|

## 6\. Model Performance Observations

* Logistic Regression achieved the highest AUC score of 0.9097 and showed good recall.
* Decision Tree provided reasonably balanced results with 84.72% accuracy.
* KNN achieved the highest accuracy of 87.18%, but its recall was relatively low.
* Naive Bayes achieved high recall but had low precision and accuracy because it predicted many sessions as purchases.
* Support Vector Machine achieved the highest F1 score of 0.6476 and MCC of 0.5805.

### Overall Best Model

Support Vector Machine was selected as the best-performing model based on its F1 score and MCC. Logistic Regression achieved a slightly higher AUC, while KNN achieved the highest accuracy.

## 7\. Streamlit Application

The application allows users to upload a test CSV, select a classification model and view evaluation metrics, confusion matrices and classification reports.

Live Streamlit Application:

https://ml-classification-assignment-2-hzjfhysxewbwtrgrh5tnv6.streamlit.app/

## 8\. Project Files

* notebooks/ML\_Assignment\_2.ipynb – Model implementation and evaluation.
* model/ – Saved trained models.
* app.py – Streamlit application.
* test\_data.csv – Test dataset.
* model\_comparison.csv – Model evaluation results.
* requirements.txt – Required Python dependencies.


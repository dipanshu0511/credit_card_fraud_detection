import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Import the data_set
credit_df = pd.read_csv("D:\Data Analytics\Email Spam Classification\creditcard.csv")

# First five rows of the dataset:
credit_df.head()

# Last five rows of the dataset:
credit_df.tail()

# Information about the dataset
credit_df.info()

# Checking for missing values in the Dataset:
credit_df.isnull().sum()

# Distribution of legit and Fraudulent transaction:
credit_df['Class'].value_counts()

# Separation of Data for Analysis

legit = credit_df[credit_df.Class == 0]
fraud = credit_df[credit_df.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistics of the dataset:
legit.Amount.describe()

fraud.Amount.describe()

# Comparison between the legit & Fraudulent Transaction:
credit_df.groupby('Class').mean()

legit_sample = legit.sample(n=492, random_state = 2)
sample_dataset = pd.concat([legit_sample, fraud], axis = 0)
sample_dataset.head()
sample_dataset.tail()

sample_dataset['Class'].value_counts()
sample_dataset.groupby('Class').mean()

X = sample_dataset.drop(columns = 'Class', axis = 1)
Y = sample_dataset['Class']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter = 1000)

# Training the logistic Regression Model with training data
model.fit(X_train, Y_train)

# Web App

st.title("Credit Card Fraud Detection System")
input_df = st.text_input("Enter the feature values:")
splitted_input = input_df.split(',')

submit = st.button("Submit")

if submit:
    # get input features
    np_df = np.asarray(splitted_input, dtype = np.float64)
    # make prediction
    prediction = model.predict(np_df.reshape(1, -1))
    # Display Result
    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")
    
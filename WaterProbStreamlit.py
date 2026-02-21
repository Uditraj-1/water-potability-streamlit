# ===============================
# Water Potability Streamlit App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

import joblib
Model = joblib.load("model.pkl")



st.title("Water Potability Prediction Dashboard")
st.write("Predict whether water is safe for drinking using Machine Learning")


# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def LoadData():
    Data = pd.read_csv("water_potability.csv")
    return Data

Data = LoadData()

st.subheader("Dataset Preview")
st.write(Data.head())


# -------------------------------
# Show Class Distribution
# -------------------------------
st.subheader("Class Distribution")
st.bar_chart(Data["Potability"].value_counts())


# -------------------------------
# Separate Features and Target
# -------------------------------
X = Data.drop("Potability", axis=1)
Y = Data["Potability"]


# -------------------------------
# Handle Missing Values
# -------------------------------
Imputer = SimpleImputer(strategy="mean")
X = Imputer.fit_transform(X)


# -------------------------------
# Train Test Split
# -------------------------------
XTrain, XTest, YTrain, YTest = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# -------------------------------
# Apply SMOTE (only on training)
# -------------------------------
Smote = SMOTE(random_state=42)
XTrain, YTrain = Smote.fit_resample(XTrain, YTrain)


# -------------------------------
# Feature Scaling
# -------------------------------
Scaler = StandardScaler()
XTrain = Scaler.fit_transform(XTrain)
XTest = Scaler.transform(XTest)


# -------------------------------
# Model Selection Sidebar
# -------------------------------
st.sidebar.header("Model Selection")

ModelName = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Decision Tree", "KNN", "SVM"]
)


# -------------------------------
# Train Selected Model
# -------------------------------
if ModelName == "Random Forest":
    Model = RandomForestClassifier(n_estimators=200, random_state=42)

elif ModelName == "Decision Tree":
    Model = DecisionTreeClassifier(random_state=42)

elif ModelName == "KNN":
    Model = KNeighborsClassifier(n_neighbors=5)

else:
    Model = SVC(probability=True)


Model.fit(XTrain, YTrain)


# -------------------------------
# Model Accuracy
# -------------------------------
YPred = Model.predict(XTest)
Accuracy = accuracy_score(YTest, YPred)

st.subheader("Model Accuracy")
st.write(f"{ModelName} Accuracy: {Accuracy:.2f}")


# -------------------------------
# User Input Section
# -------------------------------
st.sidebar.header("Enter Water Parameters")

Ph = st.sidebar.number_input("pH", 0.0, 14.0, 7.0)
Hardness = st.sidebar.number_input("Hardness", value=200.0)
Solids = st.sidebar.number_input("Solids", value=10000.0)
Chloramines = st.sidebar.number_input("Chloramines", value=7.0)
Sulfate = st.sidebar.number_input("Sulfate", value=300.0)
Conductivity = st.sidebar.number_input("Conductivity", value=400.0)
OrganicCarbon = st.sidebar.number_input("Organic Carbon", value=10.0)
Trihalomethanes = st.sidebar.number_input("Trihalomethanes", value=70.0)
Turbidity = st.sidebar.number_input("Turbidity", value=4.0)


# -------------------------------
# Create Input Array
# -------------------------------
InputData = np.array([[Ph, Hardness, Solids, Chloramines, Sulfate,
                       Conductivity, OrganicCarbon, Trihalomethanes, Turbidity]])

# Apply same preprocessing
InputData = Imputer.transform(InputData)
InputData = Scaler.transform(InputData)


# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Potability"):

    Prediction = Model.predict(InputData)[0]
    Probability = Model.predict_proba(InputData)[0][1]

    if Prediction == 1:
        st.success(f"Water is POTABLE (Probability: {Probability:.2f})")
    else:
        st.error(f"Water is NOT POTABLE (Probability: {Probability:.2f})")
        
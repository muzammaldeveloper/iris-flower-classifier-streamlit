import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

# ---- Load Data ----
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# ---- Train Model ----
model = RandomForestClassifier(random_state=42)
model.fit(df.iloc[:, :-1], df['species'])

# ---- Sidebar Inputs ----
st.sidebar.header("🌿 Input Flower Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# ---- Prediction ----
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)
predicted_species = target_names[prediction[0]]

# ---- Main Page ----
st.title("🌸 Iris Flower Prediction App")
st.markdown("This app uses **Random Forest Classifier** to predict the species of an Iris flower based on its features.")

st.subheader("🔎 Prediction Result")
st.success(f"The predicted species is: **{predicted_species}**")

# Show prediction probabilities
st.write("### 📊 Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(proba_df.T)

# ---- Data Preview ----
with st.expander("📂 See Dataset Preview"):
    st.dataframe(df.head())

# ---- Feature Importance ----
st.write("### 🌟 Feature Importance (Model Insights)")
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(df.columns[:-1], importances, color="skyblue")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)



import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Title
st.title("🏠 House Price Predictor")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('house_data.csv')

df = load_data()

# Features and target
X = df[['area', 'bedrooms', 'location']]
y = df['price']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[('location', OneHotEncoder(), ['location'])],
    remainder='passthrough'
)

# Create pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# User input
area = st.number_input("Enter area in sq ft", min_value=500, max_value=5000, step=100)
bedrooms = st.selectbox("Number of bedrooms", [1, 2, 3, 4])
location = st.selectbox("Select location", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Ahmedabad', 'Pune', 'Kolkata'])

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[area, bedrooms, location]], columns=['area', 'bedrooms', 'location'])
    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Estimateod Price: ₹{int(prediction):,}")


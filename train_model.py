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
    return pd.read_csv("house_data.csv")

df = load_data()

# Split features and target
X = df[['area', 'bedrooms', 'location']]
y = df['price']

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[('location', OneHotEncoder(), ['location'])],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])

# Train the model
model.fit(X, y)

# Input form
st.header("Enter House Details")
area = st.number_input("Area (in sq ft):", min_value=500, max_value=5000, step=100, value=1200)
bedrooms = st.selectbox("Bedrooms:", [1, 2, 3, 4])
#location = st.selectbox("Location:", ['New York', 'Los Angeles', 'Chicago'])

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[area, bedrooms, location]], columns=['area', 'bedrooms', 'location'])
    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Estimated House Price: ${int(prediction):,}")

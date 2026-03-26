import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

st.title("🏠 House Price Prediction using Mini Batch Gradient Descent")

# Load dataset
df = pd.read_csv("homeprices_banglore.csv")

X = df[['area','bedrooms']]
y = df['price']

# Train test split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled_train = scaler_x.fit_transform(x_train)
x_scaled_test = scaler_x.transform(x_test)

y_scaled_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_scaled_test = scaler_y.transform(y_test.values.reshape(-1,1))

# Build Model (TensorFlow + Keras)
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='mean_squared_error',
    metrics=['mse']
)

# Train model (Mini Batch Gradient Descent)
history = model.fit(
    x_scaled_train,
    y_scaled_train,
    epochs=100,
    batch_size=5,
    validation_data=(x_scaled_test,y_scaled_test),
    verbose=0
)

# Sidebar input
st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sqft)",1000,5000,2000)
bedrooms = st.sidebar.number_input("Bedrooms",1,5,3)

# Prediction
if st.sidebar.button("Predict Price"):

    scaled = scaler_x.transform([[area,bedrooms]])

    pred = model.predict(scaled)

    price = scaler_y.inverse_transform(pred)

    st.success(f"Predicted House Price: ₹ {price[0][0]:.2f} Lakhs")

# Cost vs Epoch Graph
st.subheader("Training Loss vs Epoch")

fig, ax = plt.subplots()

ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Mini Batch Gradient Descent")

ax.legend()

st.pyplot(fig)
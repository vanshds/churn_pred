import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("📊 Customer Churn Prediction (Re-training on Launch)")

@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, dtype=int, columns=['Geography','Gender'], drop_first=True)
    X = df.drop(columns=['Exited'])
    y = df['Exited'].values
    return X, y

X, y = load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train model
@st.cache_resource
def build_and_train():
    model = Sequential([
        Dense(11, activation='sigmoid', input_dim=X.shape[1]),
        Dense(11, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, batch_size=50, epochs=100, verbose=0, validation_split=0.2)
    return model

model = build_and_train()
st.write("✅ Model trained on app launch.")

# Sidebar inputs
st.sidebar.header("Customer Profile Input")
inputs = {
    'CreditScore': st.sidebar.number_input('Credit Score', 300, 850, 600),
    'Age': st.sidebar.number_input('Age', 18, 100, 35),
    'Tenure': st.sidebar.slider('Tenure (months)', 0, 72, 12),
    'Balance': st.sidebar.number_input('Balance', 0.0, 250000.0, 50000.0),
    'NumOfProducts': st.sidebar.selectbox('Number of Products', [1,2,3,4]),
    'HasCrCard': st.sidebar.selectbox('Has Credit Card? (1=yes)', [0,1]),
    'IsActiveMember': st.sidebar.selectbox('Active Member? (1=yes)', [0,1]),
    'EstimatedSalary': st.sidebar.number_input('Salary', 0.0, 200000.0, 50000.0),
    'Geography_Germany': st.sidebar.selectbox('Lives in Germany?', [0,1]),
    'Geography_Spain': st.sidebar.selectbox('Lives in Spain?', [0,1]),
    'Gender_Male': st.sidebar.selectbox('Gender — Male?', [0,1]),
}

if st.sidebar.button("🔍 Predict Churn"):
    X_input = pd.DataFrame([inputs])
    X_input_scaled = scaler.transform(X_input)
    prob = model.predict(X_input_scaled, verbose=0)[0, 0]
    
    # Apply your 20% threshold
    churn = prob > 0.2

    st.subheader("🚩 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2%}")
    st.success("Yes – likely to churn" if churn else "No – likely to stay")





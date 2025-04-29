import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ——— Load models & scalers ———
LB_model = load_model('LB_model.h5')
UB_model = load_model('UB_model.h5')

scaler_X_LB = joblib.load('scaler_X_LB.pkl')
scaler_y_LB = joblib.load('scaler_y_LB.pkl')

scaler_X_UB = joblib.load('scaler_X_UB.pkl')
scaler_y_UB = joblib.load('scaler_y_UB.pkl')

# ——— Streamlit UI ———
st.title('Nγ Prediction App')
st.write('Enter values for Beta, S, φ and αₕ to predict LB, UB, and average Nγ.')

st.image('structure.png', use_column_width=True)

beta = st.number_input(
    'Beta (Slope angle in degrees)',
    min_value=0.0, max_value=40.0,
    value=10.0, step=0.1
)
S = st.number_input(
    'S (Setback distance in meters)',
    min_value=0.0, max_value=3.0,
    value=0.0, step=0.1
)
phi = st.number_input(
    'φ (Soil friction angle in degrees)',
    min_value=0.0, max_value=40.0,
    value=20.0, step=0.1
)
ah = st.number_input(
    'αₕ (horizontal earthquake acceleration coefficient)',
    min_value=0.0, max_value=0.4,
    value=0.0, step=0.01
)

if beta > phi:
    st.error('⚠️ Beta must be less than or equal to φ.')
else:
    # assemble input array
    X_input = np.array([[beta, S, phi, ah]])

    # LB prediction
    X_scaled_LB = scaler_X_LB.transform(X_input)
    LB_log_scaled = LB_model.predict(X_scaled_LB)
    LB_log = scaler_y_LB.inverse_transform(LB_log_scaled)
    LB_pred = np.expm1(LB_log)[0][0]

    # UB prediction
    X_scaled_UB = scaler_X_UB.transform(X_input)
    UB_log_scaled = UB_model.predict(X_scaled_UB)
    UB_log = scaler_y_UB.inverse_transform(UB_log_scaled)
    UB_pred = np.expm1(UB_log)[0][0]

    # ensure UB ≥ LB
    if UB_pred < LB_pred:
        UB_pred = LB_pred + 0.01

    avg_pred = (LB_pred + UB_pred) / 2

    # display
    st.subheader('Predicted Nγ Values')
    st.write(f"**LB:** {LB_pred/10:.4f}")
    st.write(f"**UB:** {UB_pred/10:.4f}")
    st.write(f"**Average:** {avg_pred/10:.4f}")

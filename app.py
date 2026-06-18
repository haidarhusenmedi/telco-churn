import streamlit as st
import pickle
import pandas as pd

with open("rf_telco_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_telco.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("le_internet.pkl", "rb") as f:
    le_internet = pickle.load(f)

with open("le_contract.pkl", "rb") as f:
    le_contract = pickle.load(f)

st.set_page_config(page_title="Telco Customer Churn", layout="centered")

st.title("Telco Customer Churn Prediction")
st.write("Prediksi kemungkinan pelanggan berhenti berlangganan.")

st.sidebar.title("Telco AI")
st.sidebar.write("UAS Machine Learning")

tenure = st.number_input(
    "Lama Berlangganan (bulan)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Tagihan Bulanan ($)",
    min_value=0.0,
    value=70.0
)

internet_service = st.selectbox(
    "Jenis Internet",
    le_internet.classes_
)

contract = st.selectbox(
    "Jenis Kontrak",
    le_contract.classes_
)

if st.button("Prediksi Churn"):

    internet_encoded = le_internet.transform([internet_service])[0]
    contract_encoded = le_contract.transform([contract])[0]

    data = pd.DataFrame(
        [[tenure, monthly_charges, internet_encoded, contract_encoded]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "internet_encoded",
            "contract_encoded"
        ]
    )

    data_scaled = scaler.transform(data)

    prediksi = model.predict(data_scaled)[0]
    probabilitas = model.predict_proba(data_scaled)[0]

    st.subheader("Hasil Prediksi")

    if prediksi == 1:
        st.error(
            f"Pelanggan berpotensi CHURN ({probabilitas[1]*100:.2f}%)"
        )
    else:
        st.success(
            f"Pelanggan kemungkinan BERTAHAN ({probabilitas[0]*100:.2f}%)"
        )

    st.write(f"Churn : {probabilitas[1]*100:.2f}%")
    st.write(f"Tidak Churn : {probabilitas[0]*100:.2f}%")

    st.progress(float(probabilitas[1]))

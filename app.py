import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi Tampilan Halaman Web
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# ==========================================
# 1. MEMUAT KEMBALI MODEL & ENCODER (BACKEND)
# ==========================================
try:
    with open('rf_telco_model.pkl', 'rb') as f: 
        model_rf = pickle.load(f)
    with open('scaler_telco.pkl', 'rb') as f: 
        scaler = pickle.load(f)
    with open('le_internet.pkl', 'rb') as f: 
        le_internet = pickle.load(f)
    with open('le_contract.pkl', 'rb') as f: 
        le_contract = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Error: File model (.pkl) tidak ditemukan di repository GitHub!")
    st.stop()

# ==========================================
# 2. ANTARMUKA PENGGUNA (FRONTEND)
# ==========================================
st.title('📡 Telco Customer Churn Predictor')
st.write('Sistem AI untuk memprediksi probabilitas pengguna memutus layanan jaringan berdasarkan metrik teknis dan tagihan.')

# Syarat Tugas: Menampilkan Metrik Akurasi
st.info('**Metrik Evaluasi Model:** Random Forest Classifier | Akurasi: ~78.5%')
st.markdown("---")

st.header('Masukkan 4 Parameter Jaringan:')

# Form Input 2 Kolom
col1, col2 = st.columns(2)
with col1:
    input_tenure = st.number_input('1. Lama Berlangganan (Bulan)', min_value=0, max_value=120, value=12)
    input_charge = st.number_input('2. Tagihan Bulanan ($)', min_value=0.0, max_value=200.0, value=50.0)
    
with col2:
    input_internet = st.selectbox('3. Jenis Layanan Internet', le_internet.classes_)
    input_contract = st.selectbox('4. Jenis Kontrak', le_contract.classes_)

# ==========================================
# 3. LOGIKA PREDIKSI & OUTPUT
# ==========================================
if st.button('Hitung Probabilitas Churn 🚀'):
    # Transformasi input teks menjadi matriks angka
    internet_enc = le_internet.transform([input_internet])[0]
    contract_enc = le_contract.transform([input_contract])[0]
    
    # Susun ke dalam array 2D sesuai urutan training
    matriks_X = np.array([[input_tenure, input_charge, internet_enc, contract_enc]])
    
    # Standarisasi Z-Score
    matriks_X_scaled = scaler.transform(matriks_X)
    
    # Prediksi menggunakan Random Forest
    prediksi = model_rf.predict(matriks_X_scaled)
    
    st.markdown("---")
    st.subheader("💡 Hasil Analisis Sistem:")
    
    if prediksi[0] == 1:
        st.error("⚠️ **STATUS: BERISIKO CHURN (BERHENTI)**")
        st.write("Sistem mendeteksi probabilitas tinggi bahwa pelanggan ini akan memutus layanan. Diperlukan intervensi segera (misal: penawaran diskon atau perbaikan infrastruktur).")
    else:
        st.success("✅ **STATUS: AMAN (BERTAHAN)**")
        st.write("Profil jaringan dan tagihan menunjukkan pelanggan ini dalam kondisi stabil dan kemungkinan besar akan memperpanjang layanan.")
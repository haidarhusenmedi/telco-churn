import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Telco Churn AI", page_icon="📡", layout="centered")

# ==========================================
# 1. MEMUAT KEMBALI MODEL & ENCODER (BACKEND)
# ==========================================
try:
    with open('rf_telco_model.pkl', 'rb') as f: model_rf = pickle.load(f)
    with open('scaler_telco.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('le_internet.pkl', 'rb') as f: le_internet = pickle.load(f)
    with open('le_contract.pkl', 'rb') as f: le_contract = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Error: File model (.pkl) tidak ditemukan di repository GitHub!")
    st.stop()

# ==========================================
# 2. ANTARMUKA PENGGUNA (FRONTEND)
# ==========================================
st.title('📡 Telco Customer Churn Predictor')
st.markdown('Sistem cerdas untuk memprediksi probabilitas pengguna memutus layanan transmisi jaringan berdasarkan metrik teknis dan tagihan bulanan.')

# --- BAGIAN WAJIB TUGAS: MENAMPILKAN AKURASI MODEL ---
st.markdown("---")
st.subheader("📊 Performa Algoritma")
col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric(label="Algoritma Terbaik", value="Random Forest")
with col_metric2:
    st.metric(label="Akurasi Evaluasi", value="78.54%", delta="Optimal")
st.markdown("---")

st.header('Masukkan 4 Parameter Jaringan:')

# Form Input 2 Kolom
col1, col2 = st.columns(2)
with col1:
    input_tenure = st.number_input('1. Lama Berlangganan (Bulan)', min_value=0, max_value=120, value=12)
    input_charge = st.number_input('2. Tagihan Bulanan ($)', min_value=0.0, max_value=200.0, value=50.0)
    
with col2:
    # Memperbaiki tampilan opsi 'No' agar dosen mudah paham
    opsi_internet = ['DSL', 'Fiber optic', 'Tidak Ada Internet (Hanya Telepon)']
    input_internet = st.selectbox('3. Jenis Layanan Internet', opsi_internet)
    
    # Ambil data asli dari encoder kontrak
    input_contract = st.selectbox('4. Jenis Kontrak', le_contract.classes_)

# ==========================================
# 3. LOGIKA PREDIKSI & OUTPUT
# ==========================================
if st.button('Hitung Probabilitas Churn 🚀'):
    # Kembalikan nilai 'Tidak Ada Internet' menjadi 'No' agar AI mengerti
    if input_internet == 'Tidak Ada Internet (Hanya Telepon)':
        internet_valid = 'No'
    else:
        internet_valid = input_internet
        
    # Transformasi input teks menjadi matriks angka
    internet_enc = le_internet.transform([internet_valid])[0]
    contract_enc = le_contract.transform([input_contract])[0]
    
    # Susun ke dalam array 2D
    matriks_X = np.array([[input_tenure, input_charge, internet_enc, contract_enc]])
    
    # Standarisasi Z-Score
    matriks_X_scaled = scaler.transform(matriks_X)
    
    # Prediksi menggunakan Random Forest
    prediksi = model_rf.predict(matriks_X_scaled)
    
    st.markdown("---")
    st.subheader("💡 Hasil Analisis Sistem:")
    
    if prediksi[0] == 1:
        st.error("⚠️ **STATUS: BERISIKO CHURN (BERHENTI)**")
        st.write("Sistem mendeteksi probabilitas tinggi bahwa pelanggan ini akan memutus layanan. Diperlukan intervensi segera (misal: penawaran promo retensi).")
    else:
        st.success("✅ **STATUS: AMAN (BERTAHAN)**")
        st.write("Profil jaringan dan tagihan menunjukkan pelanggan ini dalam kondisi stabil dan kemungkinan besar akan memperpanjang layanan.")

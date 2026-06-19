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
    
    # Prediksi Kategori & Hitung Probabilitas (Keyakinan AI)
    prediksi = model_rf.predict(matriks_X_scaled)
    probabilitas = model_rf.predict_proba(matriks_X_scaled)[0] # Menghitung persentase
    
    st.markdown("---")
    st.subheader("💡 Hasil Analisis Sistem:")
    
    if prediksi[0] == 1:
        # Jika hasil Churn (Berhenti), ambil probabilitas indeks ke-1
        persentase_yakin = probabilitas[1] * 100
        st.error(f"⚠️ **STATUS: BERISIKO CHURN (BERHENTI)**")
        st.write(f"Sistem memprediksi dengan **Tingkat Keyakinan {persentase_yakin:.1f}%** bahwa pelanggan ini akan memutus layanan. Diperlukan intervensi segera (misal: penawaran promo retensi).")
        # Visualisasi Progress Bar
        st.progress(int(persentase_yakin))
    else:
        # Jika hasil Aman (Bertahan), ambil probabilitas indeks ke-0
        persentase_yakin = probabilitas[0] * 100
        st.success(f"✅ **STATUS: AMAN (BERTAHAN)**")
        st.write(f"Sistem memprediksi dengan **Tingkat Keyakinan {persentase_yakin:.1f}%** bahwa pelanggan ini akan memperpanjang layanan dan tidak pindah provider.")
        # Visualisasi Progress Bar
        st.progress(int(persentase_yakin))

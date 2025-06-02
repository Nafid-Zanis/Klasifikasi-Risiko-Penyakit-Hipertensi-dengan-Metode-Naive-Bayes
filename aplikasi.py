import pickle
import streamlit as st
import numpy as np

# Load model
hipertensi_model = pickle.load(open('model.pkl', 'rb'))

# Informasi performa model
MODEL_ACCURACY = 0.886
MODEL_AUC = 0.91

# Sidebar info model
st.sidebar.title("ℹ️ Info Model")
st.sidebar.markdown(f"""
**Akurasi Model:** {MODEL_ACCURACY * 100:.2f}%  
**ROC AUC Score:** {MODEL_AUC:.2f}  
""")

# Judul aplikasi
st.title("🩺 Klasifikasi Risiko Penyakit Hipertensi")
st.markdown("Masukkan data pasien di bawah ini untuk mengetahui risiko terkena hipertensi.")

# Form input
with st.form("form_hipertensi"):
    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Perempuan'])
        sex = 1 if sex == 'Laki-laki' else 0

        age = st.number_input('Usia (tahun)', min_value=1, max_value=120, step=1)

        currentSmoker = st.selectbox('Perokok Aktif?', options=['Ya', 'Tidak'])
        currentSmoker = 1 if currentSmoker == 'Ya' else 0

        cigsPerDay = st.number_input(
            'Batang Rokok per Hari', min_value=0, max_value=100, step=1, help="0 jika tidak merokok")

        BPMeds = st.selectbox('Menggunakan Obat Tekanan Darah?', options=['Ya', 'Tidak'])
        BPMeds = 1 if BPMeds == 'Ya' else 0

        diabetes = st.selectbox('Riwayat Diabetes?', options=['Ya', 'Tidak'])
        diabetes = 1 if diabetes == 'Ya' else 0

    with col2:
        totChol = st.number_input('Total Kolesterol (mg/dL)', min_value=50.0, max_value=400.0, format="%.1f")

        sysBP = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=70.0, max_value=300.0, format="%.1f")

        diaBP = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=40.0, max_value=200.0, format="%.1f")

        BMI = st.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=60.0, format="%.2f")

        heartRate = st.number_input('Detak Jantung (bpm)', min_value=30, max_value=200)

        glucose = st.number_input('Kadar Glukosa (mg/dL)', min_value=50.0, max_value=500.0, format="%.1f")

    # Tombol submit
    submitted = st.form_submit_button("Tes Prediksi")

    if submitted:
        try:
            input_data = [float(sex), float(age), float(currentSmoker), float(cigsPerDay),
                          float(BPMeds), float(diabetes), float(totChol), float(sysBP),
                          float(diaBP), float(BMI), float(heartRate), float(glucose)]

            # Prediksi kelas
            prediction = hipertensi_model.predict([input_data])
            
            # Probabilitas untuk semua kelas
            probabilities = hipertensi_model.predict_proba([input_data])
            
            # ✅ PERBAIKAN: Confidence adalah probabilitas maksimum
            confidence = np.max(probabilities[0]) * 100
            
            # Dapatkan kelas dengan probabilitas tertinggi
            predicted_class_index = np.argmax(probabilities[0])
            predicted_class = hipertensi_model.classes_[predicted_class_index]

            # Tentukan diagnosis berdasarkan prediksi
            if prediction[0] == 0:
                diagnosis = '✅ Pasien berisiko rendah hipertensi'
                warna = '#4CAF50'  # hijau
                saran = """
                **Saran:**
                - Pertahankan gaya hidup sehat yang sudah dijalani.
                - Tetap rutin olahraga dan konsumsi makanan bergizi.
                - Lakukan pemeriksaan kesehatan rutin setiap 6-12 bulan.
                - Hindari stres berlebihan dan istirahat yang cukup.
                - Tetap waspada terhadap faktor risiko lainnya.
                """
            else:
                diagnosis = '⚠️ Pasien berisiko tinggi hipertensi'
                warna = '#F44336'  # merah
                saran = """
                **Saran:**
                - Kurangi konsumsi garam, lemak jenuh, dan makanan olahan.
                - Rutin olahraga seperti jalan cepat 30 menit per hari.
                - Hindari merokok dan konsumsi alkohol.
                - Jaga berat badan ideal dan rutin kontrol tekanan darah.
                - Konsultasikan pengobatan ke dokter segera.
                """

            # Tampilkan hasil prediksi
            st.markdown(
                f"""
                <div style='
                    background-color:{warna};
                    padding:15px;
                    border-radius:10px;
                    color:white;
                    font-size:18px;
                    text-align:center;
                    font-weight:bold;
                '>
                    {diagnosis}<br><br>
                    Tingkat Kepercayaan: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            # Detail probabilitas semua kelas
            st.info("📊 Detail Probabilitas:")
            
            # Buat kolom untuk menampilkan probabilitas dengan lebih rapi
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                risiko_rendah_prob = probabilities[0][0] * 100
                st.metric(
                    label="Risiko Rendah", 
                    value=f"{risiko_rendah_prob:.2f}%",
                    delta=None
                )
            
            with prob_col2:
                risiko_tinggi_prob = probabilities[0][1] * 100
                st.metric(
                    label="Risiko Tinggi", 
                    value=f"{risiko_tinggi_prob:.2f}%",
                    delta=None
                )

            # Tampilkan informasi tambahan untuk debugging
            with st.expander("🔍 Detail Teknis (untuk debugging)"):
                st.write(f"**Prediksi Array:** {prediction}")
                st.write(f"**Probabilitas Raw:** {probabilities[0]}")
                st.write(f"**Kelas Model:** {hipertensi_model.classes_}")
                st.write(f"**Index Kelas Terprediksi:** {predicted_class_index}")
                st.write(f"**Kelas Terprediksi:** {predicted_class}")
                st.write(f"**Confidence Calculation:** max({probabilities[0]}) = {confidence:.4f}%")

            st.subheader("📌 Rekomendasi")
            st.markdown(saran)

            # Tambahan: Interpretasi confidence level
            st.subheader("📈 Interpretasi Tingkat Kepercayaan")
            if confidence > 90:
                confidence_level = "Sangat Tinggi"
                confidence_color = "#4CAF50"
                confidence_desc = "Model sangat yakin dengan prediksi ini."
            elif confidence > 70:
                confidence_level = "Tinggi"
                confidence_color = "#FF9800"
                confidence_desc = "Model cukup yakin dengan prediksi ini."
            elif confidence > 50:
                confidence_level = "Sedang"
                confidence_color = "#FFC107"
                confidence_desc = "Model memiliki kepercayaan sedang terhadap prediksi ini."
            else:
                confidence_level = "Rendah"
                confidence_color = "#F44336"
                confidence_desc = "Model kurang yakin dengan prediksi ini. Disarankan untuk konsultasi lebih lanjut."

            st.markdown(
                f"""
                <div style='
                    background-color:{confidence_color}20;
                    border-left: 4px solid {confidence_color};
                    padding:10px;
                    border-radius:5px;
                    margin:10px 0;
                '>
                    <strong>Tingkat Kepercayaan: {confidence_level}</strong><br>
                    {confidence_desc}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f'Terjadi kesalahan saat prediksi: {e}')
            st.error("Pastikan semua field telah diisi dengan benar.")

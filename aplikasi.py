import pickle
import streamlit as st

# Load model
hipertensi_model = pickle.load(open('model.pkl', 'rb'))

# Judul aplikasi
st.title("🩺 Klasifikasi Risiko Penyakit Hipertensi")
st.markdown(
    "Masukkan data pasien di bawah ini untuk mengetahui risiko terkena hipertensi.")

# Form input
with st.form("form_hipertensi"):
    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Perempuan'])
        sex = 1 if sex == 'Laki-laki' else 0

        age = st.number_input('Usia (tahun)', min_value=0,
                              max_value=120, step=1)

        currentSmoker = st.selectbox('Perokok Aktif?', options=['Ya', 'Tidak'])
        currentSmoker = 1 if currentSmoker == 'Ya' else 0

        cigsPerDay = st.number_input(
            'Batang Rokok per Hari', min_value=0, step=1, help="0 jika tidak merokok")

        BPMeds = st.selectbox(
            'Menggunakan Obat Tekanan Darah?', options=['Ya', 'Tidak'])
        BPMeds = 1 if BPMeds == 'Ya' else 0

        diabetes = st.selectbox('Riwayat Diabetes?', options=['Ya', 'Tidak'])
        diabetes = 1 if diabetes == 'Ya' else 0

    with col2:
        totChol = st.number_input(
            'Total Kolesterol (mg/dL)', min_value=50.0, max_value=600.0, format="%.1f")

        sysBP = st.number_input(
            'Tekanan Darah Sistolik (mmHg)', min_value=70.0, max_value=300.0, format="%.1f")

        diaBP = st.number_input(
            'Tekanan Darah Diastolik (mmHg)', min_value=40.0, max_value=200.0, format="%.1f")

        BMI = st.number_input('Body Mass Index (BMI)',
                              min_value=10.0, max_value=50.0, format="%.2f")

        heartRate = st.number_input(
            'Detak Jantung (bpm)', min_value=30, max_value=200)

        glucose = st.number_input(
            'Kadar Glukosa (mg/dL)', min_value=50.0, max_value=500.0, format="%.1f")

    # Tombol submit
    submitted = st.form_submit_button("Tes Prediksi")

    if submitted:
        try:
            input_data = [float(sex), float(age), float(currentSmoker), float(cigsPerDay),
                          float(BPMeds), float(diabetes), float(
                              totChol), float(sysBP),
                          float(diaBP), float(BMI), float(heartRate), float(glucose)]

            # Prediksi
            prediction = hipertensi_model.predict([input_data])
            probabilities = hipertensi_model.predict_proba([input_data])
            predicted_proba = probabilities[0][prediction[0]]

            # Menentukan hasil prediksi
            if prediction[0] == 0:
                diagnosis = '⚠️ Pasien terkena hipertensi risiko rendah'
                saran = """
                **Saran:**
                - Periksa kemungkinan dehidrasi atau masalah hormonal.
                - Cukupi asupan garam dan cairan sesuai anjuran dokter.
                - Hindari berdiri terlalu cepat untuk mencegah pusing.
                - Konsultasi ke tenaga medis jika tekanan rendah berulang.
                """
                warna = '#4CAF50'  # hijau
            else:
                diagnosis = '⚠️ Pasien terkena risiko hipertensi tekanan tinggi'
                saran = """
                    **Saran:**
                    - Kurangi konsumsi garam, lemak jenuh, dan makanan olahan.
                    - Rutin olahraga seperti jalan cepat 30 menit per hari.
                    - Hindari merokok dan konsumsi alkohol.
                    - Jaga berat badan ideal dan rutin kontrol tekanan darah.
                    - Konsultasikan pengobatan ke dokter.
                    """
                warna = '#F44336'  # merah

            # Menampilkan hasil diagnosis dengan probabilitas
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
                    Probabilitas: {predicted_proba:.2%}<br></br>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Tambahan info (opsional)
            st.info("Probabilitas semua kelas:")
            st.write(f"- Tekanan Rendah: {probabilities[0][0]:.2%}")
            st.write(f"- Tekanan Tinggi: {probabilities[0][1]:.2%}")

            st.subheader("📌 Rekomendasi")
            st.markdown(saran)

        except Exception as e:
            st.error(f'Terjadi kesalahan saat prediksi: {e}')

import pickle
import streamlit as st

# membaca model
hipertensi_model = pickle.load(open('model.pkl', 'rb'))

# judul web
st.title("Klasifikasi Risiko Penyakit Hipertensi")

# membagi kolom
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Perempuan'])
    sex = 1 if sex == 'Laki-laki' else 0

with col2:
    totChol = st.text_input('Input Total Kolesterol')

with col1:
    age = st.text_input('Input Usia')

with col2:
    sysBP = st.text_input('Input Tekanan Darah Sistolik')

with col1:
    currentSmoker = st.selectbox('Riwayat Merokok', options=['Ya', 'Tidak'])
    currentSmoker = 1 if currentSmoker == 'Ya' else 0

with col2:
    diaBP = st.text_input('Input Tekanan Darah Diastolik')

with col1:
    cigsPerDay = st.text_input('Input Batang Rokok per Hari')

with col2:
    BMI = st.text_input('Input Body Massa Indeks(BMI)')

with col1:
    BPMeds = st.selectbox(
        'Menggunakan Obat Tekanan Darah?', options=['Ya', 'Tidak'])
    BPMeds = 1 if BPMeds == 'Ya' else 0

with col2:
    heartRate = st.text_input('Input Detak Jantung')

with col1:
    diabetes = st.selectbox('Riwayat Diabetes?', options=['Ya', 'Tidak'])
    diabetes = 1 if diabetes == 'Ya' else 0

with col2:
    glucose = st.text_input('Input Kadar Glukosa')

# kode untuk prediksi
hipertensi_diagnosis = ''

if st.button('Tes Prediksi'):
    if '' in [age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose]:
        st.warning('Semua input numerik harus diisi ya!')
    else:
        try:
            # Input data yang sudah diubah menjadi tipe float
            input_data = [float(sex), float(age), float(currentSmoker), float(cigsPerDay),
                          float(BPMeds), float(diabetes), float(
                              totChol), float(sysBP),
                          float(diaBP), float(BMI), float(heartRate), float(glucose)]

            # Prediksi dengan model dan mendapatkan probabilitas
            hipertensi_prediction = hipertensi_model.predict([input_data])

            # Menentukan hasil diagnosis
            if hipertensi_prediction[0] == 1:
                hipertensi_diagnosis = 'Pasien terkena risiko hipertensi tekanan rendah'
            else:
                hipertensi_diagnosis = 'Pasien terkena risiko hipertensi tekanan tinggi'

            # Menampilkan hasil diagnosis dan probabilitas
            st.success(hipertensi_diagnosis)

        except Exception as e:
            st.error(f'Terjadi error saat prediksi: {e}')

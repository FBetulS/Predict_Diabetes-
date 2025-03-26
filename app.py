import streamlit as st
import joblib
import numpy as np

# Model ve scaler'ı yükle
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Diyabet Tahmin Uygulaması 🩸')
st.markdown('#### Pima Indian Diabetes Veri Seti ile Eğitilmiş Model')

# Kullanıcı girişleri için form
with st.form('diabetes_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Hamilelik Sayısı', 0, 20, 0)
        glucose = st.number_input('Glukoz (mg/dL)', 0, 200, 117)
        blood_pressure = st.number_input('Kan Basıncı (mmHg)', 0, 130, 72)
        skin_thickness = st.number_input('Cilt Kalınlığı (mm)', 0, 100, 23)
        
    with col2:
        insulin = st.number_input('İnsülin (IU/mL)', 0, 900, 30)
        bmi = st.number_input('BMI (kg/m²)', 0.0, 70.0, 32.0)
        diabetes_pedigree = st.number_input('Diyabet Soy Ağacı Fonksiyonu', 0.0, 3.0, 0.3725)
        age = st.number_input('Yaş', 0, 120, 29)
    
    submitted = st.form_submit_button('Tahmin Yap')

# Tahmin işlemi
if submitted:
    # Girdileri düzenle ve ölçeklendir
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]]).astype(float)
    scaled_data = scaler.transform(input_data)
    
    # Tahmin yap
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Sonuçları göster
    st.subheader('Sonuçlar:')
    if prediction == 1:
        st.error(f'Diyabet Riski: %{probability*100:.1f} (Yüksek Risk)')
    else:
        st.success(f'Diyabet Riski: %{probability*100:.1f} (Düşük Risk)')
    
    # Bilgilendirici mesaj
    st.info('''Not: Bu tahmin istatistiksel bir model sonucudur. 
    Kesin teşhis için lütfen bir sağlık uzmanına danışınız.''')

st.markdown('---')
st.markdown('Model Performansı (Test Seti):\n- Doğruluk: %75.8')
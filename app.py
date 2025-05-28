
# importar librerias
import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# importar modelo de random forest
modelo = joblib.load('modelo_rf_10.pkl')

st.title("Predicción de cancelaciones de Reservas")

st.write("""
### 👋 ¡Bienvenido! 

Esta aplicación te ayuda a **predecir si una reserva podría cancelarse** usando un modelo de *machine learning* entrenado con datos históricos.  
Solo necesitas ingresar algunos datos clave de la reserva (como fechas, noches, tarifa total, canal de venta, etc.) y obtendrás una predicción:  **¿Se cancelará o no?**
         
🔍 *Recuerda*: esta herramienta es solo una ayuda basada en datos históricos.  
Las predicciones no son infalibles, pero pueden ser muy útiles para tomar mejores decisiones.       
""")

st.markdown("<br>", unsafe_allow_html=True)
st.write(" 📝 Ingresa los datos de la reserva:")

col1, col2 = st.columns(2)

with col1:
# inputs no directos 
    h_res_fec_ok = st.date_input("Fecha en que se completó la reserva:", datetime.date.today())
    h_fec_lld_ok = st.date_input("Fecha de llegada al hotel:", datetime.date.today())
    h_fec_sda_ok = st.date_input("Fecha de salida del hotel:", datetime.date.today())
    h_ult_cam_fec_ok = st.date_input("Fecha del último cambio en la reserva:", datetime.date.today())

# calcular días de diferencia
dias_diferencia_cam = (h_ult_cam_fec_ok - h_res_fec_ok).days
dias_diferencia_lld = (h_fec_lld_ok - h_res_fec_ok).days
dias_diferencia_sda = (h_fec_sda_ok - h_res_fec_ok).days

# cálculo de valores con base en las fechas
ult_cam_dayofweek = h_ult_cam_fec_ok.weekday()  # valores del 0-6
sda_dayofweek = h_fec_sda_ok.weekday()  # valores del 0-6
lld_dayofweek = h_fec_lld_ok.weekday()  # valores del 0-6

with col2:
# inputs directos
    Paquete_nombre = st.selectbox("Nombre del paquete:", ["Entre semana", "Fin de semana", "Lunamielero", "Ninguno", "Sin definir", "Walk in"])  
    Clasificacion = st.selectbox("Clasificación de la reserva:", ["AFM", "AJS", "ASB", "ASD", "ASP", "AST", "GMS", "GSP", "GSU", "MJS", "MST", "|"])  
    Canal_nombre = st.selectbox("Canal de reservación:", ["Conmutador", "Directo", "Directo hotel", "Fax", "Internet", "Lada 800 Internacional", 
                                                        "Lada 800 Nacional06", "Lada 800 Nacional68", "Multivacaciones 1", "Multivacaciones 2", 
                                                        "Ninguno", "Sin definir", "Sitio propio", "Vertical Overbooking"])  
    h_tfa_total = st.number_input("Total de la tarifa:", value=0.0)

# diccionarios
paquetes = {
    "Entre semana": 0,
    "Fin de semana": 1,
    "Lunamielero": 2,
    "Ninguno": 3,
    "Sin definir": 4,
    "Walk in": 5
}
Paquete_nombre = paquetes[Paquete_nombre]

clasificaciones = {
    "AFM": 0,
    "AJS": 1,
    "ASB": 2,
    "ASD": 3,
    "ASP": 4,
    "AST": 5,
    "GMS": 6,
    "GSP": 7,
    "GSU": 8,
    "MJS": 9,
    "MST": 10,
    "|": 11
}
Clasificacion = clasificaciones[Clasificacion]

canales = {
    "Conmutador": 0,
    "Directo": 1,
    "Directo hotel": 2,
    "Fax": 3,
    "Internet": 4,
    "Lada 800 Internacional": 5,
    "Lada 800 Nacional06": 6,
    "Lada 800 Nacional68": 7,
    "Multivacaciones 1": 8,
    "Multivacaciones 2": 9,
    "Ninguno": 10,
    "Sin definir": 11,
    "Sitio propio": 12,
    "Vertical Overbooking": 13
}
Canal_nombre = canales[Canal_nombre]

st.markdown("<br>", unsafe_allow_html=True)

# predicción
if st.button("Predecir"):
        
    input_df = pd.DataFrame({
        "dias_diferencia_cam": [dias_diferencia_cam],
        "ult_cam_dayofweek": [ult_cam_dayofweek],
        "Paquete_nombre": [Paquete_nombre],
        "sda_dayofweek": [sda_dayofweek],
        "dias_diferencia_lld": [dias_diferencia_lld],
        "dias_diferencia_sda": [dias_diferencia_sda],
        "Clasificacion": [Clasificacion],
        "h_tfa_total": [h_tfa_total],
        "lld_dayofweek": [lld_dayofweek],
        "Canal_nombre": [Canal_nombre]
    })

    # predicción binaria
    pred = modelo.predict(input_df)[0]

    # probabilidad de cancelación (clase 1)
    prob = modelo.predict_proba(input_df)[0][1]  

    if pred == 1:
        st.error(f"🚨 Esta reserva tiene una alta probabilidad de ser cancelada.")
    else:
        st.success(f"✅ Esta reserva probablemente no será cancelada.")

    st.info(f"🔢 Probabilidad estimada de cancelación: **{prob:.2%}**")

    st.markdown("<br>", unsafe_allow_html=True)


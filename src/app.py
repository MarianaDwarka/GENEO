import os
import time
import base64
import streamlit as st
import pandas as pd
from models.model_phase1 import prompt_db_vector, consult_db
from ga.ga_phase1 import GAtelco
from gif.create_gif import create_gif

# Configuración de la página
st.set_page_config(
    page_title="GENEO",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilos personalizados
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 4rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .description {
            font-size: 1.2rem;
            line-height: 1.6;
        }
        .stButton button {
            background-color: #1E88E5;
            color: white;
            font-size: 1.2rem;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("G.E.N.E.O 📡 ")


st.markdown(
    """GENEO is designed as an application function (**AF**) based on **LLM + GA** for **planning management** in the insertion 
    of **Edge UPFs** and their **resource optimization**.
    """, unsafe_allow_html=True)


# ----------------------------- #
# SECCIÓN 1: PLANIFICACIÓN DE RED
# ----------------------------- #

st.subheader("Network Planning")
st.markdown(
    """The **network planning** phase determines the optimal placement of **User Plane Functions (UPFs)** 
    in a 5G edge computing environment. The goal is to minimize latency and maximize service quality by optimizing 
    UPF locations using **genetic algorithms (GA)**.
    """, unsafe_allow_html=True)

st.markdown(
    """
    - **Genetic Algorithm Optimization:** UPF positions are selected based on population demand and network constraints.  
    - **Adaptive Infrastructure Deployment:** The algorithm continuously evolves to improve network efficiency.  
    """, unsafe_allow_html=True)

st.markdown("**Define the number of UPFs to be deployed in the network:**", unsafe_allow_html=True)
upf_number = st.number_input("Quantity of UPFs: ", 1, 50, step=1)

# Contenedor para mostrar los resultados en una tabla
result_container = st.empty()

if st.button("Optimize Network"):
    with st.spinner("Running Genetic Algorithm..."):
        # Instancia del algoritmo genético
        ga_instance = GAtelco(
            generations=5, 
            router=upf_number, 
            mu=0.8, 
            eta=0.35, 
            people_priority={"tipo1": 5000, "tipo2": 15000, "tipo3": 100000}
        )
        
        # Ejecutar el algoritmo genético una sola vez
        resultado_ga = ga_instance.GA()

        # Extraer resultados de todas las generaciones
        generaciones = list(range(1, len(resultado_ga['imagen']) + 1))  # Rango de generaciones
        aptitudes = resultado_ga['imagen']  # Lista de aptitudes
        dominios = [str(d) for d in resultado_ga['dominio']]  # Convertir los dominios a texto

        # Crear un DataFrame con los resultados
        df_results = pd.DataFrame({
            "Generation": generaciones,
            "Aptitude": aptitudes,
            "Domain": dominios
        })
        
        # Mostrar la tabla en Streamlit
        result_container.dataframe(df_results)

    st.success("Optimization completed!")

st.divider()
st.divider()


# ----------------------------- #
# SECCIÓN 2: OPTIMIZACIÓN DE RED
# ----------------------------- #

st.subheader("Network Optimization")
st.markdown(
    """ The **Optimization stage** performs the **reallocation of network resources** according to the **capacities in the Edge UPFs**, considering 
      the **mobility of the users for each hour**.
    - A **Large Language Model (LLM)** is used to validate the optimization results, ensuring compliance with **5G technical specifications**.
    - Once validated, the **LLM** also generates the **configuration change command** for the Edge UPFs.
    - This is executed using **API operations (AsSessionWithQoS)** defined by **3GPP**.
    """, unsafe_allow_html=True)

# Mostrar el GIF con la evolución de la optimización
gif_path = "media/plots.gif"

# Esperar hasta que el archivo GIF sea generado
with st.spinner("Generating Optimization GIF..."):
    while not os.path.exists(gif_path):
        st.warning("The optimization GIF is being generated... Please wait.")
        time.sleep(2)

# Leer y convertir el GIF a base64 para mostrarlo en Streamlit
with open(gif_path, "rb") as file_:
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")

# Mostrar el GIF en Streamlit
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="Network Optimization Evolution">',
    unsafe_allow_html=True
)

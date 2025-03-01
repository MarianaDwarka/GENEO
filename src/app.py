import os
import time
import base64
import streamlit as st
import pandas as pd
from utils import calculate_tco
from models.model_phase1 import prompt_db_vector, consult_db
from ga.ga_phase1 import GAtelco
from gif.create_gif import create_gif

# Configuración de la página
st.set_page_config(
    page_title="GENEO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Menú lateral con botones
path_icon = 'media/wireless.webp'
st.logo(path_icon, icon_image=path_icon)#, size="large")

st.sidebar.title("Menu")
if st.sidebar.button("Network Planning"):
    st.session_state["page"] = "Network Planning"
if st.sidebar.button("Network Optimization"):
    st.session_state["page"] = "Network Optimization"

# Establecer la página por defecto si no existe en session_state
if "page" not in st.session_state:
    st.session_state["page"] = "Network Planning"

option = st.session_state["page"]

st.title("G.E.N.E.O 📡 ")

st.markdown(
    """ #### Genetic Evolution for Network Efficiency and Optimization
    """, unsafe_allow_html=True)

st.markdown(
    """GENEO is designed as an application function (**AF**) based on **LLM + GA** for **planning management** in the insertion 
    of **Edge UPFs** and their **resource optimization**.
    """, unsafe_allow_html=True)

# ----------------------------- #
# SECCIÓN 1: PLANIFICACIÓN DE RED
# ----------------------------- #


if option == "Network Planning":
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
    
    # Mostrar información sobre los UPFs y sus costos
    st.markdown(
        """
        #### UPF Pricing Information
        
        The network optimization considers the following pricing structure for UPFs based on their total bandwidth capacity:
        
        | Capacity (Small) | Cost (Small) | Capacity (Medium) | Cost (Medium) | Capacity (Large) | Cost (Large) |
        |-----------------|-------------|-----------------|-------------|-----------------|-------------|
        | 70 Gbps        | $19,000     | 140 Gbps       | $34,000     | 300 Gbps       | $48,000     |

        The algorithm will determine the optimal UPF locations and calculate the Total Cost Optimization (TCO) based on the selected configurations.
        """,
        unsafe_allow_html=True)

    
    # Definir la cantidad de UPFs
    st.markdown("**Define the number of UPFs to be deployed in the network:**", unsafe_allow_html=True)
    upf_number = st.number_input("Quantity of UPFs: ", 1, 50, step=1)
    
    result_container = st.empty()
    tco_container = st.empty()
    message_container = st.empty()
    upf_info_container = st.empty()
    
    if st.button("Run Network Planning"):
        with st.spinner("Running Genetic Algorithm..."):
            ga_instance = GAtelco(
                generations=5, 
                router=upf_number, 
                mu=0.75, 
                eta=0.5, 
                people_priority={"tipo1": 5000, "tipo2": 15000, "tipo3": 100000}
            )
            resultado_ga = ga_instance.GA()
            last_iteration = resultado_ga['dominio'][-1]
            total_cost = calculate_tco(last_iteration)
            
            df_results = pd.DataFrame({
                "Generation": list(range(1, len(resultado_ga['imagen']) + 1)),
                "Aptitude": resultado_ga['imagen'],
                "Domain": [str(d) for d in resultado_ga['dominio']]
            })
            result_container.dataframe(df_results)
            
            num_upfs = upf_number
            capacities = last_iteration[2::3]
            latitudes = last_iteration[0::3]
            longitudes = last_iteration[1::3]
            
            tco_container.markdown(
                f"""
                <div style="
                    padding: 20px;
                    background-color: #1E88E5;
                    color: white;
                    text-align: center;
                    border-radius: 10px;
                    font-size: 24px;
                    font-weight: bold;
                ">
                    ✅ TCO: ${total_cost:,.0f} USD
                </div>
                """,
                unsafe_allow_html=True
            )
            
            message_container.markdown("**Selected UPFs Information**")
            df_upfs = pd.DataFrame({
                "UPF #": range(1, num_upfs + 1),
                "Latitude": latitudes,
                "Longitude": longitudes,
                "Capacity (Gbps)": capacities
            })
            upf_info_container.dataframe(df_upfs)
        
        st.success("Planning completed!")

# ----------------------------- #
# SECCIÓN 2: OPTIMIZACIÓN DE RED
# ----------------------------- #

elif option == "Network Optimization":
    st.subheader("Network Optimization")
    st.markdown(
        """ The **Optimization stage** performs the **reallocation of network resources** according to the **capacities in the Edge UPFs**, considering 
          the **mobility of the users for each hour**.
        - A **Large Language Model (LLM)** is used to validate the optimization results, ensuring compliance with **5G technical specifications**.
        - Once validated, the **LLM** also generates the **configuration change command** for the Edge UPFs.
        - This is executed using **API operations (AsSessionWithQoS)** defined by **3GPP**.
        """, unsafe_allow_html=True)
    
    gif_path = "media/plots.gif"
    with st.spinner("Generating Optimization GIF..."):
        while not os.path.exists(gif_path):
            st.warning("The optimization GIF is being generated... Please wait.")
            time.sleep(2)
    
    with open(gif_path, "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
    
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Network Optimization Evolution">',
        unsafe_allow_html=True
    )

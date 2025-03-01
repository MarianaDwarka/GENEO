import os
import json
from pinecone import Pinecone  # Importa Pinecone para gestionar bases de datos vectoriales
from dotenv import load_dotenv  # Carga variables de entorno desde un archivo .env
from langchain_openai import OpenAIEmbeddings  # Modelo de embeddings de OpenAI
from langchain_community.chat_models import ChatOpenAI  # Modelo de lenguaje de OpenAI
from langchain.vectorstores import Pinecone as pinelang  # Almacén de vectores usando Pinecone
from langchain.vectorstores.utils import DistanceStrategy  # Estrategias de distancia para búsqueda de similitud

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Configuración de las variables de entorno para OpenAI y Pinecone
os.environ["OPENAI_API_TYPE"] = "openai"
os.environ["MODEL_OPENAI_4O"] = "gpt-4o"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Clave de API de OpenAI
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")  # Clave de API de Pinecone

# Prompt utilizado para la consulta a la base de datos basada en estándares ITU-T
prompt_db_vector = """
Based on ITU standards (specifically ITU-T Y.1541 and ITU-T G.1010), I need a bandwidth distribution for different use case priorities.
The information provided is as follows:
1. Total available bandwidth: {total_bandwidth}.
2. Number of priorities: 3.
3. Number of users per priority:
- Priority 1 (High): {p1_num}.
- Priority 2 (Medium): {p2_num}.
- Priority 3 (Low): {p3_num}.
The use cases for each priority are as follows:
- Priority 1 (High): Real-time applications like VoIP and video conferencing.
- Priority 2 (Medium): Interactive applications like video streaming and telework.
- Priority 3 (Low): Non-critical applications like web browsing and downloads.
Based on ITU standards, what would be the optimal bandwidth distribution for each priority, considering the total bandwidth, number of priorities, and number of users per priority? Provide a clear explanation of how the distribution is calculated and justify the percentages assigned to each priority.
"""

prompt_interpretacion_planeacion = """"Consulta en la base de conocimientos vectorizada con los estándares 3GPP TS 23.501 y 23.502: C
                                    omparar los anchos de banda asignados a los UPFs (BW_UPF_1, BW_UPF_2, ..., BW_UPF_N) con los requisitos
                                    recomendados para los User Plane Functions (UPFs) en un escenario Edge de una red 5G Core. Si todos los anchos
                                    de banda asignados están alineados con lo recomendado, devolver 'OK'. Si alguno no está alineado, devolver
                                    'NOT OK' y aumentar el ancho de banda de esos UPFs en una cantidad 'X'.
                                    A continuación de presento los resultaods obtenidos:
                                        - ubicacion de los upfs {}
                                        - estas son sus capacidades {}
                                        - esta es la latencia que resultado de los upfs optimizados {}
                                        - todos los upfs son de Huawei
                                    Con esto me puedes decir si esta 'OK' o 'NOT OK'

                                    """

# Respuesta modelo basada en los estándares ITU-T para la distribución de ancho de banda
answere = """
Basado en los estándares ITU-T Y.1541 e ITU-T G.1010, la distribución de ancho de banda para las prioridades dadas sería la siguiente:
    1. **Prioridad 1 (Alta) - VoIP y Videoconferencias**:
        - **Porcentaje de ancho de banda**: 40%.
        - **Justificación**: Las aplicaciones en tiempo real requieren baja latencia y un ancho de banda garantizado para asegurar una calidad de servicio (QoS) óptima. Según ITU-T Y.1541, el tráfico en tiempo real debe tener la máxima prioridad.
    2. **Prioridad 2 (Media) - Streaming y Teletrabajo**:
        - **Porcentaje de ancho de banda**: 35%.
        - **Justificación**: Las aplicaciones interactivas necesitan un ancho de banda suficiente para evitar buffering y retrasos, pero no requieren la misma prioridad que las aplicaciones en tiempo real. ITU-T G.1010 recomienda asignar un ancho de banda moderado para este tipo de tráfico.
    3. **Prioridad 3 (Baja) - Navegación Web y Descargas**:
        - **Porcentaje de ancho de banda**: 25%.
        - **Justificación**: Las aplicaciones no críticas pueden funcionar con un ancho de banda limitado sin afectar significativamente la experiencia del usuario. Este tipo de tráfico tiene la menor prioridad según los estándares ITU.
    **Cálculo de la Distribución**:
    - **Ancho de banda total**: {response}.
    - **Ancho de banda para Prioridad 1**: 40% del total.
    - **Ancho de banda para Prioridad 2**: 35% del total.
    - **Ancho de banda para Prioridad 3**: 25% del total.
Esta distribución asegura que las aplicaciones críticas reciban el ancho de banda necesario mientras se optimiza el uso de los recursos disponibles.
"""

# Se instancia el modelo de embeddings de OpenAI para transformar texto en vectores
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Función para mejorar la respuesta generada, integrando información del algoritmo genético y estándares ITU-T
def improve_response(original_response: str, pos, latency, bw_total):
    apikey = os.environ["OPENAI_API_KEY"]

    # Prompt que complementa la respuesta inicial con información sobre latencia y optimización con algoritmos genéticos
    complement_info = """
    You are a network expert with extensive experience in latency analysis and communication system performance. Your goal is to provide a response that integrates two sources of information:
    1. Knowledge base: Technical explanation of average network latency, including factors that influence it and its importance in communication system performance.
    2. Genetic algorithm: Analysis of average latency of a network population, using selection, crossover, and mutation techniques to optimize latency parameters.
    Develop a response that:
    - Explains the concept of latency in detail from the knowledge base
    - Describes how the genetic algorithm can model and optimize average latency
    - Integrates the genetic algorithm results with the theoretical explanation from the knowledge base
    - Provides insights on how to improve latency in different network scenarios

    Prioritize clarity in the explanation, showing how the two information sources {conocimiento} and the latency obtained from the genetic algorithm {latencia} complement each other to offer a deeper understanding of network latency. The router location found by the genetic algorithm was the following {position} in latitude and longitude.

    Use the following information to complement the response:
    Based on ITU-T Y.1541 and ITU-T G.1010 standards, the bandwidth distribution for the given priorities would be as follows:
        - **Total Bandwidth**: {bandwidth}.
        - **Bandwidth for Priority 1**: 40% of total.
        - **Bandwidth for Priority 2**: 35% of total.
        - **Bandwidth for Priority 3**: 25% of total.
    This distribution ensures that critical applications receive the necessary bandwidth while optimizing the use of available resources.
    """.format_map({
        "conocimiento": original_response,
        "latencia": "{} ms".format(latency),
        "position": str(pos),
        "bandwidth": "{} Gbps".format(bw_total)
    })

    # Se instancia el modelo GPT-4o de OpenAI con una temperatura baja para generar respuestas consistentes
    llm = ChatOpenAI(model="gpt-4o", temperature=0.16, api_key=apikey)
    chain = llm.invoke(complement_info)  # Se invoca el modelo con la información generada
    return chain

# Función para consultar la base de datos vectorial en Pinecone y mejorar la respuesta con el modelo GPT-4o
def consult_db(question, pos, latency, bw_total):
    # Conexión con la base de datos de Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "hackathon"  # Nombre del índice en Pinecone
    index = pc.Index(index_name)

    # Creación del almacén de vectores para realizar búsquedas por similitud
    vectorstore = pinelang(
        index,
        embeddings_model.embed_query,
        "text",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    # Búsqueda en la base de datos vectorial utilizando Pinecone
    response = vectorstore.similarity_search(question, namespace="connectivity", k=10)

    print(response[0])
    exit()

    # Mejora de la respuesta obtenida con el modelo de lenguaje
    improved_response = improve_response(response[0].page_content, pos, latency, bw_total)

    return improved_response.content  # Se devuelve la respuesta mejorada

consult_db(prompt_interpretacion_planeacion, pos=[],latency=[], bw_total=[])

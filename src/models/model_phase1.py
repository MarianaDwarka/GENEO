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
                                        - ubicacion de los upfs estan en una matriz donde la primera columna son las latitudes y la segunda son las longitudes {}
                                        - estas son sus capacidades {}
                                        - todos los upfs son de Huawei
                                    """

# Se instancia el modelo de embeddings de OpenAI para transformar texto en vectores
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Función para mejorar la respuesta generada, integrando información del algoritmo genético y estándares ITU-T
def improve_response(original_response, carga_hora):
    apikey = os.environ["OPENAI_API_KEY"]

    # Prompt que complementa la respuesta inicial con información sobre latencia y optimización con algoritmos genéticos
    complement_info = """{} con esta informacion {} me puedes decir si esta "OK" o "NOT OK" la distribución de cargas de los upfs
    se claro y conciso con la explicación. """.format(original_response,carga_hora)

    # Se instancia el modelo GPT-4o de OpenAI con una temperatura baja para generar respuestas consistentes
    llm = ChatOpenAI(model="gpt-4o", temperature=0.16, api_key=apikey)
    chain = llm.invoke(complement_info)  # Se invoca el modelo con la información generada
    return chain

# Función para consultar la base de datos vectorial en Pinecone y mejorar la respuesta con el modelo GPT-4o
def consult_db(question, pos, bw_total, carga_optima):
    # Conexión con la base de datos de Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("INDEX_PINECONE_PLN")  # Nombre del índice en Pinecone
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

    # Mejora de la respuesta obtenida con el modelo de lenguaje
    improved_response = improve_response(response[0].page_content, carga_hora=carga_optima )

    return improved_response.content  # Se devuelve la respuesta mejorada

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


prompt_json = """
Consulta en la base de conocimientos vectorizada con los estándares 3GPP TS 23.501, 23.502, 29.122, 29.517 y 29.571, y
los archivos YAML TS29122_AsSessionWithQoS.yaml, TS29122_MonitoringEvent.yaml y TS29571_CommonData.yaml: a partir de lo anterior,
tu tarea es generar un JSON con la estructura de comando POST para la operación AsSessionWithQoS, incluyendo los parámetros necesarios
para configurar los anchos de banda de los UPFs. Los nombres de los UPFs son UPF_1, UPF_2, ..., UPF_n; las IPs de los UPFs son UPF_1_IP, UPF_2_IP, ..., UPF_n_IP;
y los anchos de banda a configurar son {}. El JSON debe seguir las especificaciones técnicas y la estructura definida en
los estándares y archivos YAML mencionados.
"""

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Función para mejorar la respuesta generada, integrando información del algoritmo genético y estándares ITU-T
def improve_response_json(carga_hora):
    apikey = os.environ["OPENAI_API_KEY"]

    # Prompt que complementa la respuesta inicial con información sobre latencia y optimización con algoritmos genéticos
    complement_info = """
    Esta es la estructura del json que debes usar de generar con la capacidad de los los upfs {} \n""".format(carga_hora)+\
    """
    {
      "operation": "AsSessionWithQoS",
      "UPFs": [
        {
          "name": "UPF_1",
          "ip": "UPF_1_IP",
          "bandwidth": "UPF_1_bw"
        },
        {
          "name": "UPF_2",
          "ip": "UPF_2_IP",
          "bandwidth": "UPF_2_bw"
        },
        {
          "name": "UPF_3",
          "ip": "UPF_3_IP",
          "bandwidth": "UPF_3_bw"
        }
        ...
      ]
    }

    esa es una ejemplo de un json pero necesito que tu generes un json usando la cantidad de upfs, solo necesito que regreses el json unicamente

    """
    # Se instancia el modelo GPT-4o de OpenAI con una temperatura baja para generar respuestas consistentes
    llm = ChatOpenAI(model="gpt-4o", temperature=0.16, api_key=apikey)
    chain = llm.invoke(complement_info)  # Se invoca el modelo con la información generada
    return chain

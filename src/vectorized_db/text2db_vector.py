import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Cargar las variables de entorno desde el archivo .env
dotenv_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))  # Un nivel arriba
load_dotenv(dotenv_path)

# Obtener variables de entorno
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("NAMESPACE_PINECONE")  # Asegúrate de que sea el entorno correcto
#index_name = os.getenv("INDEX_PINECONE_PLN") # Índice para planeación
index_name = os.getenv("INDEX_PINECONE_OPT") # Índice para optimización
openai_api_key = os.getenv("LLAMA_API_KEY")  # Suponiendo que esta es la clave de OpenAI

# Inicializar Pinecone con la nueva versión de la API
pc = Pinecone(api_key=pinecone_api_key)

# Verificar si el índice ya existe
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Creando el índice '{index_name}' en Pinecone...")
    pc.create_index(
        name=index_name,
        dimension=3072,  # Ajustar según el tamaño del embedding
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Ajustar según región de tu proyecto
    )
else:
    print(f"El índice '{index_name}' ya existe.")

# Conectar al índice
index = pc.Index(index_name)
print(f"Índice '{index_name}' listo para usar.")


embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))

def upload_dbvector():
    """
    Carga documentos Markdown en una base de datos vectorial en Pinecone.
    """
    # Define la carpeta donde se encuentran los archivos Markdown
    path_folder_main = "../markdown_files/"

    # Verificar si la carpeta existe
    if not os.path.exists(path_folder_main):
        print(f"Error: La carpeta '{path_folder_main}' no existe.")
        return
    
    folder_main = os.listdir(path_folder_main)

    # Obtener configuración del namespace en Pinecone
    namespace = os.getenv("NAMESPACE_PINECONE")

    # Iterar sobre las carpetas en el directorio principal
    for folder in folder_main:
        folder_path = os.path.join(path_folder_main, folder)
        if not os.path.isdir(folder_path):
            continue  # Saltar si no es una carpeta
        
        #files = os.listdir(folder_path)
        files = [f for f in os.listdir(folder_path) if not f.startswith(".")]  # Ignorar archivos ocultos
        print(f"Procesando carpeta: {folder}")

        # Iterar sobre los archivos en la carpeta actual
        for file in files:
            file_path = os.path.join(folder_path, file)

            print(f"Procesando archivo: {file}")

            # Cargar contenido del archivo Markdown
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()

            # Dividir el documento en fragmentos
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            doc = text_splitter.split_documents(documents)

            # Agregar metadatos a cada fragmento
            for text in doc:
                text.metadata["source"] = f"documents/{folder}_{file}.md"
                text.metadata["archivo"] = folder
                text.metadata["page"] = file.split(".")[0].split("-")[-1]
                text.metadata["type_document"] = "norm"

            # Almacenar los fragmentos en Pinecone
            PineconeVectorStore.from_documents(
                documents=doc,
                index_name=index_name,
                embedding=embeddings_model,
                namespace=namespace
            )

            print(f"Documento procesado: {folder}/{file}")

            # Pausa de 1 segundo para evitar sobrecarga
            time.sleep(1)

        # Pausa adicional de 1 segundo antes de procesar la siguiente carpeta
        time.sleep(1)


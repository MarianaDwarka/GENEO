# Importación de módulos necesarios
import os  # Para la manipulación de archivos y directorios
import shutil  # Para eliminar carpetas de manera recursiva
from dotenv import load_dotenv  # Para cargar variables de entorno
from PyPDF2 import PdfWriter, PdfReader  # Clases específicas para escribir y leer PDFs
from llama_parse import LlamaParse  # Para convertir PDFs en texto Markdown


# Cargar el archivo .env desde una ruta específica
dotenv_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))  # Un nivel arriba
load_dotenv(dotenv_path)

# Configuración del parser para convertir PDFs en Markdown
def configure_parser():
    """
    Configura y devuelve una instancia del parser para convertir PDFs a Markdown.
    """
    return LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),  # Obtiene la API Key desde el .env
        result_type="markdown",
        #premium_mode=True,
        num_workers=4,
        verbose=True,
        language="es",
    )

parser = configure_parser()

# Función para verificar y crear una carpeta si no existe
def check_folder(name_folder: str, delete=False):
    """
    Verifica si existe una carpeta con el nombre dado y la crea si no existe.
    Si delete=True, elimina la carpeta existente y la vuelve a crear.

    Parámetros:
    name_folder (str): Nombre de la carpeta a verificar o crear.
    delete (bool): Si es True, elimina la carpeta antes de crearla nuevamente.
    """
    a = os.getcwd().replace('\\', '/')  # Obtiene la ruta actual y normaliza los separadores
    path = f"{a}/{name_folder}"  # Construye la ruta completa de la carpeta

    try:
        os.makedirs(path)  # Intenta crear la carpeta
        print("folder created")
    except FileExistsError:  # Si ya existe, maneja la excepción
        print("folder already exists")
        if delete:
            try:
                shutil.rmtree(path)  # Elimina la carpeta y su contenido
            except OSError:
                os.remove(path)  # Si no puede eliminarla recursivamente, la elimina directamente


# Función para dividir un PDF en páginas individuales
def split_pdf(name: str, folder: str):
    """
    Divide un archivo PDF en múltiples archivos PDF, uno por cada página.

    Parámetros:
    name (str): Ruta del archivo PDF de entrada.
    folder (str): Nombre de la carpeta donde se guardarán los PDFs de cada página.
    """
    pdf_file1 = PdfReader(open(name, 'rb'))  # Abre el PDF en modo lectura
    num_page = len(pdf_file1.pages)  # Obtiene el número total de páginas
    path = os.path.join(os.getcwd(), folder)  # Construye la ruta destino
    check_folder(folder)  # Verifica y crea la carpeta si es necesario

    # Itera sobre cada página del PDF
    for page_num in range(num_page):
        pdf_writer = PdfWriter()  # Crea un objeto para escribir PDFs
        page = pdf_file1.pages[page_num]  # Obtiene la página actual
        pdf_writer.add_page(page)  # Agrega la página al nuevo archivo PDF
        
        # Guarda la página como un nuevo archivo PDF individual
        output_path = os.path.join(path, f"{page_num}.pdf")
        with open(output_path, "wb") as outputStream:
            pdf_writer.write(outputStream)

    print("se ha terminado de procesar con éxito el archivo" + name )


# Función para procesar una página PDF y convertirla en Markdown
def process_page(page: str, path_md: str, path_page: str = None):
    """
    Convierte una página PDF en un archivo de texto Markdown.

    Parámetros:
    page (str): Nombre del archivo PDF de la página.
    path_md (str): Ruta donde se guardará el archivo Markdown.
    path_page (str): Ruta donde se encuentra el PDF original.
    """

    name_page = page.split('.pdf')[0]  # Obtiene el nombre del archivo sin la extensión
    full_path = os.path.join(path_page, page)  # Construye la ruta completa del archivo PDF
    markdown_content = parser.load_data(full_path)  # Convierte el PDF a Markdown
    #print(markdown_content)
    text_markdown_content = markdown_content[0].dict().get('text')  # Extrae el contenido en texto

    new_folder = "markdown_files"
    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", new_folder))  # Obtiene el path absoluto correcto
    path = os.path.join(base_path, path_md)  # Concatena correctamente la subcarpeta

    os.makedirs(path, exist_ok=True) 

    # Guarda el contenido extraído en un archivo Markdown
    with open(os.path.join(path, f"{name_page}.md"), 'w', encoding='utf-8') as f:
        f.write(text_markdown_content)


# Función principal que convierte PDFs en Markdown
def main_pdf2md(path="../standards"):
    """
    Convierte todos los archivos PDF de una carpeta en archivos Markdown.

    Parámetros:
    path (str): Ruta de la carpeta que contiene los archivos PDF.
    """
    l = os.listdir(path)  # Lista todos los archivos en la carpeta de pdf

    # Filtrar solo los archivos .pdf
    pdf_files = [file for file in l if file.lower().endswith(".pdf")]

    if not pdf_files:
        print("No se encontraron archivos PDF en la carpeta.")
        return
    
    # Itera sobre cada archivo en la carpeta
    for file in pdf_files:
        folder_name = file.split(".pdf")[0]  # Obtiene el nombre base del archivo sin extensión
        print(folder_name)  # Muestra el nombre del archivo procesado

        path_page = os.path.join("../pdf_files", folder_name)

        # Crea una carpeta para almacenar los PDFs divididos
        os.makedirs(path_page, exist_ok=True)
        
        # Divide el PDF en archivos individuales por página
        split_pdf(os.path.join(path, file), path_page)
        
        # Procesa cada página divida y la convierte en Markdown
        for f in os.listdir(path_page):
            process_page(f, folder_name, path_page=path_page)


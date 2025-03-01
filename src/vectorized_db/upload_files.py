from task import main_pdf2md
from text2db_vector import upload_dbvector


#Divide los pdf y los convierte a markdown para la parte de planeación
#main_pdf2md(path="../standards/Planning") 

#Divide los pdf y los convierte a markdown para la parte de optimización
#main_pdf2md(path="../standards/Optimization") 

#Sube los markdowns a pinecone
upload_dbvector() 

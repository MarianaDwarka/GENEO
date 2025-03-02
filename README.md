# GENEO: Genetic Evolution for Network Efficiency and Optimization


![PANDORA Cover](geneo_portada.jpg)


## Overview
G.E.N.E.O is a Application Function (AF) based on Genetic Algorithm (GA) supervised with Large Language Model (LLM), focused on planning, optimization, and configuration of Edge computing in 5G SA (Standalone) scenarios.

## Project Structure and Usage
* **`app.py`:** The main entry point. Run `streamlit run app.py` to start the application.
* **`orquestador.py`:** Contains the genetic algorithm implementation.
* **`model_phase1.py`:** Contains the llm implementation.
* **`upload_files.py`:** Used for uploading and processing PDF files.
* **`images_gif`:** Stores generated images from the genetic algorithm.
* **`requirements.txt`:** Lists necessary Python packages.
* **`.env`:** Stores API keys for OpenAI, Pinecone, and other services.

## Workflow
1. **PDF Processing:**
   * `upload_files.py` processes PDF files, splitting them and converting them to Markdown format using Llama Index.
   * The converted text is vectorized using the `text-embedding-3-large` model and stored in a Pinecone vector database.
2. **Genetic Algorithm:**
   * The `orquestador.py` script executes the genetics algorithms.
   * The algorithm iteratively evolves a population of solutions, optimizing router placements.
   * The algorithm queries the Pinecone database using OpenAI's GPT-4 to retrieve relevant standards and guidelines for network planning.
3. **Visualization:**
   * The algorithm generates visualizations of the optimized network, saving them to the `images_gif` folder.

## Prerequisites
* Python environment with required packages (specified in `requirements.txt`).
* API keys for OpenAI, Pinecone, and other services (stored in `.env`).
* A Pinecone vector database.

## Installation
1. Clone the repository.
2. Create a `.env` file and add your API keys.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

## Contributions

- Luis Ulloa Barriga
- Fernando Moreno Gomez
- Mariana Carmona Cruz
- Rodolfo Lagunas Jardines
- Juan Carlos Montero Vilchis

## Additional Notes
* **Similarity Search:** The project employs cosine similarity to compare vector representations of the processed text and the query.
* **Image Generation:** The genetic algorithm produces visual representations of the network topology.
* **Customization:** The algorithm can be customized by modifying the genetic operators, fitness function, and other parameters.


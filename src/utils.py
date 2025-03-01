import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from random import random, choice
import os

def plot_map_lazy_frame(df_in: pl.LazyFrame, nombre: str):
    """Visualiza la distribución geográfica de puntos con diferentes prioridades."""
    df_5000 = df_in.filter(pl.col("prioridad")==1).collect()
    df_15000 = df_in.filter(pl.col("prioridad")==2).collect()
    df_100000 = df_in.filter(pl.col("prioridad")==3).collect()
    
    plt.figure(figsize=(15, 10))
    plt.scatter(df_100000["SW_LONG"], df_100000["SW_LAT"], color="red", label="Prioridad_3", marker=".", edgecolor="black")
    plt.scatter(df_15000["SW_LONG"], df_15000["SW_LAT"], color="blue", label="Prioridad_2", marker=".", edgecolor="black")
    plt.scatter(df_5000["SW_LONG"], df_5000["SW_LAT"], color="yellow", label="Prioridad_1", marker=".", edgecolor="black")
    
    plt.title(f"Distribución - Hora {nombre.split('_')[1].split('.')[0]}")
    plt.legend()
    
    plt.savefig(os.path.join("gif_p", nombre))
    plt.close()

def plot_map(df_in):
    """Visualiza la distribución geográfica de puntos con diferentes prioridades.
    
    Esta función genera un gráfico de dispersión que muestra la ubicación geográfica
    de puntos clasificados en tres niveles de prioridad, cada uno representado con
    un color diferente.
    
    Args:
        df_in (pandas.DataFrame): DataFrame que contiene las columnas 'SW_LAT' y 'SW_LONG'
                                 con las coordenadas de latitud y longitud de los puntos.
    """
    df_5000 = df_in.iloc[:5000]
    df_15000 = df_in.iloc[5000:20000]
    df_100000 = df_in.iloc[20000:120000]
    plt.figure(figsize=(15, 10))
    plt.scatter(df_100000["SW_LONG"], df_100000["SW_LAT"], color="red", label = "Prioridad_3", marker=".", edgecolor="black")
    plt.scatter(df_15000["SW_LONG"], df_15000["SW_LAT"], color="blue", label="Prioridad_2", marker=".", edgecolor="black")
    plt.scatter(df_5000["SW_LONG"], df_5000["SW_LAT"], color="yellow", label="Prioridad_1", marker=".", edgecolor="black")
    plt.legend()
    plt.show()

def dataframe(df_t):
    """Asigna atributos de prioridad, ancho de banda y latencia a un DataFrame.
    
    Esta función toma un DataFrame y lo enriquece con tres nuevas columnas:
    'prioridad', 'ancho_de_banda(Mbps)' y 'latencia(ms)', asignando valores
    específicos a cada segmento del DataFrame según su índice.
    
    Args:
        df_t (pandas.DataFrame): DataFrame al que se añadirán las nuevas columnas.
                                Se espera que tenga 120000 filas.
    
    Returns:
        pandas.DataFrame: El DataFrame original con las tres nuevas columnas añadidas.
    """
    df_t["prioridad"] = [0]*120000
    df_t.loc[:5001, "prioridad"] = 1
    df_t.loc[5001:20001, "prioridad"] = 2
    df_t.loc[20001:, "prioridad"] = 3
    df_t["ancho_de_banda(Mbps)"] = ["0"]*120000
    df_t.loc[:5001, "ancho_de_banda(Mbps)"] = "500-1000"
    df_t.loc[5001:20001, "ancho_de_banda(Mbps)"] = "100-500"
    df_t.loc[20001:, "ancho_de_banda(Mbps)"] = "25-100"
    df_t["latencia(ms)"] = [0]*120000
    df_t.loc[:5001, "latencia(ms)"] = 15
    df_t.loc[5001:20001, "latencia(ms)"] = 40
    df_t.loc[20001:, "latencia(ms)"] = 100
    return df_t

def make_dfs(df_m):
    """Genera 24 archivos CSV con coordenadas ligeramente modificadas.
    
    Esta función simula el movimiento de puntos a lo largo de 24 horas (o periodos),
    aplicando pequeñas modificaciones aleatorias a las coordenadas de latitud y longitud.
    Para cada hora, guarda un archivo CSV con las coordenadas actualizadas.
    
    Args:
        df_m (pandas.DataFrame): DataFrame que contiene las columnas 'SW_LAT' y 'SW_LONG'
                                con las coordenadas de latitud y longitud que serán modificadas.
    """
    rdm_n = [1,-1]
    n = len(df_m)
    for h in range(24):
        lat_modifications = 0.09 * np.array([random()*choice(rdm_n) for _ in range(n)])
        long_modifications = 0.09 * np.array([random()*choice(rdm_n) for _ in range(n)])
        
        # Aplicar las modificaciones
        df_m["SW_LAT"] = df_m["SW_LAT"] + lat_modifications
        df_m["SW_LONG"] = df_m["SW_LONG"] + long_modifications
        
        df_m.to_csv(f'Hora_{h}.csv', index=False) if h>9 else df_m.to_csv(f'Hora_0{h}.csv', index=False)
        
def calculate_distances(df, point_lat, point_long, lat_col='SW_LAT', long_col='SW_LONG'):
    """
    Calcula la distancia en kilómetros entre cada punto de un DataFrame y un punto específico.
    
    Esta función utiliza la fórmula de Haversine para calcular distancias precisas sobre
    la superficie terrestre teniendo en cuenta la curvatura de la Tierra.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene las coordenadas de latitud y longitud.
        point_lat (float): Latitud del punto de referencia (en grados decimales).
        point_long (float): Longitud del punto de referencia (en grados decimales).
        lat_col (str, opcional): Nombre de la columna de latitud en el DataFrame. Por defecto 'SW_LAT'.
        long_col (str, opcional): Nombre de la columna de longitud en el DataFrame. Por defecto 'SW_LONG'.
    
    Returns:
        pandas.DataFrame: El DataFrame original con una columna adicional 'distancia_km' que
                         contiene la distancia en kilómetros desde cada punto al punto de referencia.
    
    Ejemplo:
        >>> df = pd.DataFrame({'SW_LAT': [40.7128, 34.0522], 'SW_LONG': [-74.0060, -118.2437]})
        >>> result = calculate_distances(df, 51.5074, -0.1278)
        >>> print(result)
            SW_LAT   SW_LONG  distancia_km
            0  40.7128  -74.0060   5570.222180
            1  34.0522 -118.2437   8755.602341
    """
    result_df = df.copy()
    
    lat1 = np.radians(point_lat)
    lon1 = np.radians(point_long)
    lat2 = np.radians(df[lat_col])
    lon2 = np.radians(df[long_col])
    
    R = 6371.0
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    result_df['distancia_km'] = distance
    
    return result_df

def calculate_tco(last_iteration: np.ndarray) -> int:
    """
    Calcula el Costo Total de Propiedad (TCO) de los UPFs basándose en su capacidad.
    
    Parámetros:
    - last_iteration (np.ndarray): Array con las coordenadas y capacidad de los UPFs.

    Retorna:
    - int: El costo total en USD.
    """
    
    # Diccionario de precios por capacidad en Gbps
    prices = {
        70: 19000,   # Small
        140: 34000,  # Medium
        300: 48000   # Large
    }

    # Extraer solo las capacidades (tercera columna en cada grupo de 3 valores)
    capacities = last_iteration[2::3]  # Tomar cada tercer valor a partir del índice 2

    # Calcular el costo total sumando los precios correspondientes a cada capacidad
    total_cost = sum(prices.get(int(cap), 0) for cap in capacities)

    return total_cost

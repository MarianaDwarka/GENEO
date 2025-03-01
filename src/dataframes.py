import numpy as np
import pandas as pd
from utils import gen_pts
import matplotlib.pyplot as plt

points = [
    (80.7128, -74.0060), 
    (4.0522, -138.2437), 
    (151.5074, -0.1278) ]

longitudes, latitudes = gen_pts(points)

df = pd.DataFrame({
    'user': list(range(1,len(longitudes)+1)),
    'longitudes': longitudes,
    'latitudes': latitudes
})

df_m = df.sample(frac=1).reset_index(drop=True)

df_m["Prioridad"] = [0]*120000
df_m["ancho_de_banda(Mbps)"] = [0]*120000
df_m["latencia(ms)"] = [0]*120000


df_m.loc[:5001, "Prioridad"] = 1
df_m.loc[5001:20001, "Prioridad"] = 2
df_m.loc[20001:, "Prioridad"] = 3

df_m.loc[:5001, "ancho_de_banda(Mbps)"] = "500-1000"
df_m.loc[5001:20001, "ancho_de_banda(Mbps)"] = "100-500"
df_m.loc[20001:, "ancho_de_banda(Mbps)"] = "25-100"

df_m.loc[:5001, "latencia(ms)"] = "1-10"
df_m.loc[5001:20001, "latencia(ms)"] = "10-30"
df_m.loc[20001:, "latencia(ms)"] = "30-100"

df_5000 = df_m.iloc[:5000]
df_15000 = df_m.iloc[5000:20000]
df_100000 = df_m.iloc[20000:120000]

plt.figure(figsize=(15, 10))
plt.scatter(df_100000["longitudes"], df_100000["latitudes"], color="green", label = "Prioridad_3", marker=".")
plt.scatter(df_15000["longitudes"], df_15000["latitudes"], color="blue", label="Prioridad_2", marker=".")
plt.scatter(df_5000["longitudes"], df_5000["latitudes"], color="red", label="Prioridad_1", marker=".")
plt.legend()
plt.title("Distribución inicial")
plt.show()
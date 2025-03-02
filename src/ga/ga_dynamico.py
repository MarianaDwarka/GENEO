import os
import numpy as np
import random as rd
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Función para realizar cruce de padres en un algoritmo genético
def offspring(parents: np.ndarray) -> np.ndarray:
    """
    Realiza el cruce de dos individuos (padres) para generar dos nuevos individuos (hijos).
    """
    babys = np.zeros(parents.shape)  # Inicializa la matriz de descendientes
    split = np.random.randint(0, parents.shape[1])  # Punto de cruce aleatorio
    # Intercambio de segmentos entre los padres para generar los hijos
    babys[0, :split] = parents[0, :split]
    babys[0, split:] = parents[1, split:]
    babys[1, :split] = parents[1, :split]
    babys[1, split:] = parents[0, split:]
    return babys


def calculate_distance(upf_ubicacion:np.ndarray, usuarios)->tuple:
    usuarios = usuarios[["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].values
    mask_upf = np.where(np.arange(upf_ubicacion.size) % 3 != 2)[0]
    mask_capacidad = np.where(np.arange(upf_ubicacion.size) % 3 == 2)[0]
    rows = mask_upf.size//2
    distances = np.zeros((usuarios.shape[0], mask_capacidad.size))
    upf_matrix = upf_ubicacion[mask_upf].reshape(rows,2)
    for col, upf in enumerate(upf_matrix):
        d = np.sqrt(np.sum(np.power(usuarios[:,:2] - upf, 2), axis=1))
        distances[:,col] = d
    return np.argsort(distances),usuarios[:,2], upf_ubicacion[mask_capacidad]


def usuarios_dataset(filename:str="../csvs/Hora_00_MEX_v2.csv"):
    df = pd.read_csv(filename)
    #rango_tipo = {1:[0.4,0.6], 2:[0.1,0.2], 3:[0.001,0.02]}
    usuario = {}
    for tipo in range(1,4):
        #s = df[df["prioridad"]==tipo]["ancho_de_banda(Gbps)"].size
        #df[df["prioridad"]==tipo]["ancho_de_banda(Gbps)"] = np.random.uniform(rango_tipo[tipo][0],rango_tipo[tipo][1],s)
        usuario["tipo{}".format(tipo)]=df[df["prioridad"]==tipo][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==tipo])*.1))
    # usuario = {
    #     "tipo1": df[df["prioridad"]==1][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==1])*.1)), #Convertir en array,
    #     "tipo2": df[df["prioridad"]==2][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==2])*.1)),
    #     "tipo3": df[df["prioridad"]==3][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==3])*.1))
    # }
    lat, long, banda = [], [], []
    for tipo in usuario:
        lat.extend(usuario[tipo]["SW_LAT"].tolist())
        long.extend(usuario[tipo]["SW_LONG"].tolist())
        banda.extend(usuario[tipo]["ancho_de_banda(Gbps)"].tolist())
    df_tmp = {"SW_LAT":lat, "SW_LONG":long, "ancho_de_banda(Gbps)":banda}
    return pd.DataFrame(df_tmp)


class GAdynamic:
    """
    Clase que implementa un Algoritmo Genético para optimizar la ubicación de routers
    en función de la latencia para diferentes tipos de usuarios.
    """
    num_routers = 1  # Número de routers a optimizar
    dimension=3
    def __init__(self, 
                 upf_planeacion:np.ndarray,
                dataframe_hour:pd.DataFrame,
                router:int,
                mu=0.75,
                eta=0.25,
                hora_actual: int = 1,
                generations: int = 50,
                people_priority: dict = {"tipo1": 5000, "tipo2": 15000, "tipo3": 100000},
                pop_size: int = 100):
        """
        Inicializa los parámetros del algoritmo genético.
        """
        self.hora_actual = hora_actual
        self.generations = generations  # Número de generaciones
        self.mu = mu  # Probabilidad de cruce
        self.eta = eta  # Probabilidad de mutación
        self.pop_size = pop_size  # Tamaño de la población
        # Número de usuarios de cada tipo de prioridad
        self.num_user_p1 = people_priority["tipo1"]
        self.num_user_p2 = people_priority["tipo2"]
        self.num_user_p3 = people_priority["tipo3"]
        self.router = router
        self.mask = np.where(np.arange(router*3) % 3 != 2)[0]
        self.distancias_sort,self.capacidad_usuario, self.capacidad_upf = calculate_distance(upf_ubicacion=upf_planeacion,
                                                        usuarios=dataframe_hour)

    def population(self):
        """
        Genera una población inicial aleatoria de porcentajes de capacidad.
        """
        porcentaje_capacidad = np.random.uniform(low=0, high=1, size=(self.pop_size, self.router))
        return porcentaje_capacidad

    def mutation(self, pop_tmp, ind_optimo, index_optimo):
        """
        Aplica mutación a algunos individuos de la población.
        """
        eta_mat = np.random.random(self.pop_size)
        mut_eta = np.where(eta_mat <= self.eta)[0]
        count_mod = mut_eta.size ## esta variable determina cuales individuos se van a mutar
        pop_tmp[mut_eta, :] = np.random.uniform(low=0, high=1, size=(count_mod, self.router))
        pop_tmp[index_optimo, :] = ind_optimo
        return pop_tmp

    def selection(self, pop_tmp: np.ndarray, l_eval: list) -> dict:
        """
        Selecciona los individuos más aptos de la población usando torneo.
        """
        rows, cols = pop_tmp.shape
        winners = np.zeros(rows // 2, dtype=int)
        eval_pop = np.array(l_eval)
        father = np.zeros((rows // 2, cols))
        list_pop = np.arange(0, 100).reshape(50, 2)
        np.random.shuffle(list_pop)
        i = 0
        while i < 50:
            row = list_pop[i, :]
            index = np.where(eval_pop[row] == np.max(eval_pop[row]))[0][0]
            father[i, :] = pop_tmp[row[index], :]
            np.put(winners, i, row[index])
            i += 1
        list_pop = list_pop.flatten()
        list_pop = np.delete(np.sort(list_pop), winners, None)
        mother = pop_tmp[list_pop, :]
        return {'father': father, 'mother': mother}


    def cross(self, pop_gen: np.ndarray, best: np.ndarray, eval_pop) -> np.ndarray:
        """
        Realiza el cruce entre individuos seleccionados.
        """
        pop_tmp = pop_gen.copy()
        new_pop = np.zeros(pop_gen.shape)
        men_woman = self.selection(pop_tmp=pop_tmp, l_eval=eval_pop)
        father, mother = men_woman['father'], men_woman['mother']
        r_u, r_d = 0, 2
        for f, m in zip(father, mother):
            parents = np.matrix([f, m])
            if rd.random() < self.mu:
                new_pop[r_u:r_d, :] = offspring(parents=parents)
            else:
                new_pop[r_u:r_d, :] = parents
            r_u = r_d
            r_d += 2
        new_pop[0, :] = best  # Mantener el mejor individuo
        return new_pop


    def fx_(self, pop_tmp):
        row, col = self.distancias_sort.shape
        personas_sin_upf = np.zeros(pop_tmp.shape[0])
        n_personas = row
        n_upf = col
        capacidades_upf = self.capacidad_upf
        for k,capacidad_porcentual in enumerate(pop_tmp):
            ocupacion_upf = np.zeros(n_upf, dtype=int)
            asignaciones = [-1] * n_personas
            for i in range(n_personas):
                # Obtener índices de casillas ordenadas por distancia para esta persona
                indices_casillas_ordenadas = self.distancias_sort[i]
                # Intentar asignar a la casilla más cercana que tenga capacidad
                asignada = False
                for idx_upf in indices_casillas_ordenadas:
                    if ocupacion_upf[idx_upf] < capacidades_upf[idx_upf]*capacidad_porcentual[idx_upf]:
                        # Asignar persona a esta casilla
                        asignaciones[i] = idx_upf
                        ocupacion_upf[idx_upf] += 1
                        asignada = True
                        break
                # Si no se pudo asignar, se queda como -1 (sin casilla)
            # Contar personas sin casilla
            #personas_sin_upf = asignaciones.count(-1)
            personas_sin_upf[k] = asignaciones.count(-1)
        return personas_sin_upf

    def fx(self, pop_tmp):
        row, col = self.distancias_sort.shape
        n_pop = pop_tmp.shape[0]
        personas_sin_upf = np.zeros(n_pop)
        n_personas = row
        n_upf = col
        capacidades_upf = self.capacidad_upf

        for k, capacidad_porcentual in enumerate(pop_tmp):
            # Pre-calcular capacidades efectivas
            capacidades_efectivas = capacidades_upf * capacidad_porcentual

            ocupacion_upf = np.zeros(n_upf, dtype=int)
            asignaciones = np.full(n_personas, -1)

            for i in range(n_personas):
                indices_casillas_ordenadas = self.distancias_sort[i]

                # Vectorizar la comparación de capacidades disponibles
                casillas_disponibles = indices_casillas_ordenadas[
                    ocupacion_upf[indices_casillas_ordenadas] < capacidades_efectivas[indices_casillas_ordenadas]
                ]

                if len(casillas_disponibles) > 0:
                    idx_upf = casillas_disponibles[0]  # La más cercana disponible
                    asignaciones[i] = idx_upf
                    ocupacion_upf[idx_upf] += 1

            # Contar personas sin asignación de forma vectorizada
            personas_sin_upf[k] = np.sum(asignaciones == -1)

        return personas_sin_upf

    def get_fitness(self, pop_gen: np.ndarray) -> tuple:
        """
        Evalúa la aptitud de cada individuo en la población basada en la latencia.

        Parámetros:
        - pop_gen: np.ndarray -> Matriz de población de routers.

        Retorna:
        - eval: np.ndarray -> Latencias de todos los individuos.
        - best_ind: np.ndarray -> Individuo con menor latencia.
        - fitness: float -> Valor mínimo de latencia.
        - fitness_tipo1: float -> Latencia mínima para usuarios tipo 1.
        """
        eval = self.fx(pop_tmp=pop_gen)
        #fitness_tipo1_index = np.argmin(eval_tipo1).item()  # Índice del mejor individuo en tipo 1.
        fitness_index = np.argmin(eval).item()  # Índice del mejor individuo en latencia total.
        best_ind = pop_gen[fitness_index]  # Selección del mejor individuo.
        fitness = eval[fitness_index]  # Mejor valor de latencia total.
        #fitness_tipo1 = eval_tipo1[fitness_tipo1_index]  # Mejor latencia para usuarios tipo 1.
        return eval, best_ind, fitness,fitness_index  # Retorna las métricas de aptitud.

    def GA(self) -> dict:
        """
        Ejecuta el Algoritmo Genético para encontrar la mejor ubicación de routers.

        Retorna:
        - Un diccionario con la mejor ubicación de routers ('dominio') y la evolución de la latencia ('imagen').
        """
        df_result = {'generacion': np.arange(self.generations), 'optimal': [], 'avg': []}
        # col_position = np.where(np.arange(self.router*3) % 3 != 2)[0]
        #routers_optimos = dominio[-1][col_position].reshape(self.router,2)
        # Inicialización de matrices de resultados.
        imagen = np.zeros(self.generations)
        pop_avg = np.zeros(self.generations)
        dominio = np.zeros((self.generations, self.router))
        #imagen_tipo1 = np.zeros(self.generations)
        print(f"Generación   \t     | Aptitud óptimo   \t       | Dominio        ")
        # Generar población inicial y evaluar aptitud.
        pop = self.population()
        eval, best_ind, fitness, fitness_index = self.get_fitness(pop_gen=pop)
        # Almacenar la mejor solución inicial.
        dominio[0, :] = best_ind
        imagen[0] = fitness
        #imagen_tipo1[0] = fit_tipo1
        pop_avg[0] = np.mean(eval)
        # Evolución del algoritmo genético.
        for generation in range(1, self.generations):
            # Aplicación de cruce y mutación.
            pop_tmp = self.cross(pop_gen=pop, best=best_ind, eval_pop=eval)
            pop_tmp = self.mutation(pop_tmp=pop_tmp, ind_optimo=best_ind, index_optimo=fitness_index)
            pop = pop_tmp.copy()

            # Evaluar aptitud después de la evolución.
            eval, best_ind, fitness, fitness_index = self.get_fitness(pop_gen=pop)
            pop_avg[generation] = np.mean(eval)

            # Almacenar la mejor solución de la generación.
            if fitness < imagen[generation - 1]:  # Si mejora la solución, actualizar valores.
                imagen[generation] = fitness
                #imagen_tipo1[generation] = fit_tipo1
                dominio[generation, :] = best_ind
            else:  # Si no mejora, mantener la mejor solución anterior.
                imagen[generation] = imagen[generation - 1]
                #imagen_tipo1[generation] = imagen_tipo1[generation - 1]
                dominio[generation, :] = dominio[generation - 1]

            # Imprimir estado de la generación actual.
            print(f"{generation} \t | {imagen[generation]} \t | {dominio[generation]}")
            if imagen[generation]==0:
                return {'dominio': dominio, 'imagen': imagen}  # Retorna las mejores soluciones encontradas.

        # Almacenar resultados finales.
        dominio_str = [str(d) for d in dominio]
        df_result['optimal'] = imagen
        #df_result['optimal_tipo1'] = imagen_tipo1
        df_result['avg'] = pop_avg
        df_result["dominio"] = dominio_str
        df_result = pd.DataFrame(df_result)

        # Generar gráficos de evolución.
        # self.plot_routers(df_result, dominio)
        #self.plot_optimal(df_result, dominio)

        # Guardar resultados en un archivo CSV.
        df_result.to_csv(f"./save_csv/resultado_{self.generations}_{self.hora_actual}_dinamico.csv", index=False)

        return {'dominio': dominio, 'imagen': imagen}  # Retorna las mejores soluciones encontradas.

#x = GAdynamic(router=5).GA()
#pop_ = x.population()
#print(x)

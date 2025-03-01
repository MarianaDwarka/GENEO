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

def put_capacitie(matrix:np.ndarray, rows:int)->np.ndarray:
    matrix_copy = matrix.copy()
    col_capacitie = np.where(np.arange(matrix_copy.shape[1])%3==2)[0]
    for nf in col_capacitie:
        matrix_copy[:,nf] = np.random.choice([70,140,300],rows)
    return matrix_copy

def init_people(type_machine="mac"):
    """
    Inicializa las ubicaciones de los usuarios con una distribución normal.
    """
    csv_route = "./csvs/Hora_00_MEX_v2.csv" if type_machine=="mac" else  r".\csv's\Hora_00_MEX_v2.csv" #
    df = pd.read_csv(csv_route)
    rango_tipo = {1:[0.4,0.6], 2:[0.1,0.2], 3:[0.001,0.02]}
    usuario = {}
    for tipo in range(1,4):
        s = df[df["prioridad"]==tipo]["ancho_de_banda(Gbps)"].size
        df[df["prioridad"]==tipo]["ancho_de_banda(Gbps)"] = np.random.uniform(rango_tipo[tipo][0],rango_tipo[tipo][1],s)
        usuario["tipo{}".format(tipo)]=np.array(df[df["prioridad"]==tipo][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==tipo])*.1)))
    # usuario = {
    #     "tipo1": np.array(df[df["prioridad"]==1][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==1])*.1))), #Convertir en array,
    #     "tipo2": np.array(df[df["prioridad"]==2][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==2])*.1))),
    #     "tipo3": np.array(df[df["prioridad"]==3][["SW_LAT", "SW_LONG", "ancho_de_banda(Gbps)"]].sample(int(len(df[df["prioridad"]==3])*.1)))
    # }
    return usuario



class GAtelco:
    """
    Clase que implementa un Algoritmo Genético para optimizar la ubicación de routers
    en función de la latencia para diferentes tipos de usuarios.
    """
    num_routers = 1  # Número de routers a optimizar
    dimension=3
    def __init__(self, mu=0.75, eta=0.5, generations: int = 1000,
                 pop_size: int = 100,
                 router=1):
        """
        Inicializa los parámetros del algoritmo genético.
        """
        self.generations = generations  # Número de generaciones
        self.mu = mu  # Probabilidad de cruce
        self.eta = eta  # Probabilidad de mutación
        self.pop_size = pop_size  # Tamaño de la población
        # Número de usuarios de cada tipo de prioridad
        self.router = router
        self.mask = np.where(np.arange(self.router*3) % 3 != 2)[0]
        self.distribution_people = init_people()

    # def init_people(self):
    #     """
    #     Inicializa las ubicaciones de los usuarios con una distribución normal.
    #     """
    #     usuario = {
    #         "tipo1": np.array([np.random.normal(150, 50, self.dimension) for _ in range(self.num_user_p1)]),
    #         "tipo2": np.array([np.random.normal(170, 100, self.dimension) for _ in range(self.num_user_p2)]),
    #         "tipo3": np.array([np.random.normal(210, 160, self.dimension) for _ in range(self.num_user_p3)])
    #         #"tipo3": np.array([np.random.uniform(30, 300, self.dimension) for _ in range(self.num_user_p3)])
    #     }
    #     rangos_prioridad = {"tipo1":[1, 2,self.num_user_p1],
    #                         "tipo2":[0.4, 0.8, self.num_user_p2],
    #                         "tipo3":[0.01, 0.3, self.num_user_p3]}
    #     for tipo, val in rangos_prioridad.items():
    #         usuario[tipo][:,2] = np.random.uniform(low=val[0],high=val[1],size=val[2])
    #     return usuario

    #distribution_people = init_people()

    def population(self):
        """
        Genera una población inicial aleatoria de routers.
        """
        nfs = np.random.uniform(low=0, high=1500, size=(self.pop_size, 3*self.router))
        # (latitud, longitud)
        mask_latitud = np.where(self.mask%3==0)[0]
        mask_longitud = np.where(self.mask%3!=0)[0]
        rango_latitud = np.random.uniform(low=13.889853705541531, high=33.36265536391364, size=(self.pop_size, mask_latitud.size))
        rango_longitud = np.random.uniform(low=-117.80190998938004, high=-85.96884640258124, size=(self.pop_size, mask_longitud.size))
        nfs[:,self.mask[mask_latitud]] = rango_latitud
        nfs[:,self.mask[mask_longitud]] = rango_longitud
        nfs_capacitie = put_capacitie(nfs, self.pop_size)
        del nfs, mask_latitud, mask_longitud, rango_latitud, rango_longitud
        return nfs_capacitie


    def selection_tmp(self, pop_tmp: np.ndarray, l_eval: list) -> dict:
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

    def selection(self, pop_tmp: np.ndarray, l_eval: list) -> dict:
        """
        Selecciona los individuos más aptos de la población usando torneo.
        Versión optimizada con operaciones vectorizadas.
        """
        rows, cols = pop_tmp.shape
        half_rows = rows // 2
        eval_pop = np.array(l_eval)

        # Crear parejas para torneos de forma más eficiente
        indices = np.arange(rows)
        np.random.shuffle(indices)
        tournament_pairs = indices.reshape(half_rows, 2)

        # Determinar ganadores de cada torneo de manera vectorizada
        # Para cada par, seleccionar el índice con mejor evaluación
        pair_evals = eval_pop[tournament_pairs]
        winner_mask = np.argmax(pair_evals, axis=1)

        # Seleccionar los índices ganadores de cada par
        winners = np.array([tournament_pairs[i, idx] for i, idx in enumerate(winner_mask)])

        # Seleccionar padres directamente
        father = pop_tmp[winners]

        # Seleccionar todas las filas excepto los ganadores como madres
        losers = np.setdiff1d(indices, winners)
        mother = pop_tmp[losers]

        return {'father': father, 'mother': mother}

    def cross_tmp(self, pop_gen: np.ndarray, best: np.ndarray, eval_pop) -> np.ndarray:
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

    def cross(self, pop_gen: np.ndarray, best: np.ndarray, eval_pop) -> np.ndarray:
        """
        Realiza el cruce entre individuos seleccionados de manera optimizada.
        """
        # Crear nueva población y realizar selección
        new_pop = np.zeros(pop_gen.shape)
        men_woman = self.selection(pop_tmp=pop_gen.copy(), l_eval=eval_pop)
        father, mother = men_woman['father'], men_woman['mother']

        # Generar todos los números aleatorios de una vez para determinar si hay cruce
        cross_mask = np.random.random(len(father)) < self.mu

        # Agrupar padres para procesamiento vectorizado
        parents_array = np.stack([father, mother], axis=1)

        # Calcular índices de filas en la población final
        row_indices = np.arange(0, len(father) * 2, 2)

        # Llenar la nueva población
        for i, (should_cross, parents) in enumerate(zip(cross_mask, parents_array)):
            row_start = row_indices[i]
            row_end = row_start + 2

            if should_cross:
                # Realizar cruce
                new_pop[row_start:row_end, :] = offspring(parents=np.matrix(parents))
            else:
                # Mantener padres originales
                new_pop[row_start:row_end, :] = parents

        # Mantener el mejor individuo (elitismo)
        new_pop[0, :] = best

        return new_pop

    def mutation(self, pop_tmp, ind_optimo, index_optimo):
        """
        Aplica mutación a algunos individuos de la población.
        """
        mask_latitud = np.where(self.mask%3==0)[0]
        mask_longitud = np.where(self.mask%3!=0)[0]
        eta_mat = np.random.random(self.pop_size)
        mut_eta = np.where(eta_mat <= self.eta)[0]
        rango_latitud = np.random.uniform(low=13.889853705541531, high=33.36265536391364, size=(mut_eta.size, mask_latitud.size))
        rango_longitud = np.random.uniform(low=-117.80190998938004, high=-85.96884640258124, size=(mut_eta.size, mask_longitud.size))
        mut_temporal = np.zeros((mut_eta.size,3*self.router))
        count_mod = mut_eta.size ## esta variable determina cuales individuos se van a mutar
        mut_temporal[:, self.mask[mask_latitud]] = rango_latitud
        mut_temporal[:, self.mask[mask_longitud]] = rango_longitud
        pop_tmp[mut_eta, :] = mut_temporal
        pop_tmp[mut_eta, :] = put_capacitie(pop_tmp[mut_eta, :], count_mod)
        pop_tmp[index_optimo, :] = ind_optimo
        return pop_tmp

    def fx_multiple_optimized_(self, pop_tmp: np.ndarray) -> np.ndarray:
        c = 2 / (10 ** 6)  # Factor de conversión de distancia a latencia.
        L_total = np.zeros(self.pop_size)  # Inicialización del arreglo para latencia total.
        distribution_people = self.distribution_people  # Inicializa la distribución de usuarios.
        ponderacion = np.array([80, 10, 10])  # Ponderaciones para cada tipo de usuario como array numpy
        col_position = np.where(np.arange(pop_tmp.shape[1]) % 3 == 2)[0]

        # Precomputamos valores constantes
        prioridades = list(distribution_people.keys())
        limites_superiores = {"tipo1": 15, "tipo2": 50, "tipo3": 100}
        penalidades = {"tipo1": 100, "tipo2": 30, "tipo3": 10}

        # Vectorizamos las asignaciones
        asignaciones_router = {}

        for k, row in enumerate(pop_tmp):
            asignaciones_router[k] = []
            capacidad_tipo = np.zeros(self.router)
            L = np.zeros(3)  # Arreglo para almacenar latencias por tipo de usuario.
            for i, prioridad in enumerate(prioridades):
                usuarios = distribution_people[prioridad]
                lim_sup, penality = limites_superiores[prioridad], penalidades[prioridad]
                # Precalculamos todas las distancias de una vez
                distances_router = np.zeros((usuarios.shape[0], col_position.size))

                # Extraer las coordenadas de los routers de una vez
                router_coords = row[self.mask].reshape(self.router,2)
                # Cálculo vectorizado de distancias para todos los routers a la vez
                for j, coords in enumerate(router_coords):
                    # Expandir dimensiones para broadcast
                    router_pos = coords.reshape(1, 2)
                    user_pos = usuarios[:, :2]

                    # Cálculo vectorizado de distancias euclidianas
                    d = np.sqrt(np.sum(np.power(user_pos - router_pos, 2), axis=1))
                    l_tmp = c * d

                    # Aplicar penalizaciones vectorizadamente
                    l_tmp = np.where(l_tmp > lim_sup, l_tmp + penality, l_tmp)
                    distances_router[:, j] = l_tmp

                # Encontrar el router más cercano para cada usuario
                router_cercano = np.argmin(distances_router, axis=1)
                asignaciones_router[k].append(router_cercano)

                # Cálculo de latencia optimizado
                if i == 0:
                    # Para tipo1, usamos el máximo
                    L[i] = ponderacion[i] * np.max(np.min(distances_router, axis=1)) / 100
                else:
                    # Para los demás tipos
                    L[i] = ponderacion[i] * np.sum(distances_router[np.arange(len(router_cercano)), router_cercano]) / 100
                for router in range(self.router):
                    usuario_asig_router = np.where(asignaciones_router[k][i]==router)[0]
                    capacidad_tipo[router] = row[router] - np.sum(distribution_people[prioridad][usuario_asig_router,2])
            # Penalización por capacidad excedida
            penalizacion_capacidad = 10 if np.any(capacidad_tipo < 0) else 0
            L_total[k] = np.mean(L) + penalizacion_capacidad

        return L_total

    def fx_multiple_optimized(self, pop_tmp: np.ndarray) -> np.ndarray:
        c = 2 / (10 ** 6)  # Factor de conversión de distancia a latencia
        L_total = np.zeros(self.pop_size)  # Arreglo para latencia total

        # Precomputar valores constantes (mover fuera de la función si es posible)
        prioridades = list(self.distribution_people.keys())
        limites_superiores = np.array([15, 50, 100])  # Valores para tipo1, tipo2, tipo3
        penalidades = np.array([100, 30, 10])  # Valores para tipo1, tipo2, tipo3
        ponderacion = np.array([80, 10, 10])

        for k, row in enumerate(pop_tmp):
            capacidad_tipo = np.zeros(self.router)
            L = np.zeros(3)  # Latencias por tipo de usuario

            # Extraer coordenadas de routers una sola vez
            router_coords = row[self.mask].reshape(self.router, 2)

            for i, prioridad in enumerate(prioridades):
                usuarios = self.distribution_people[prioridad]
                user_pos = usuarios[:, :2]

                # Cálculo vectorizado de distancias para todos los usuarios y routers a la vez
                # Reshapear para broadcasting: usuarios (n_users, 1, 2), routers (1, n_routers, 2)
                distances = np.sqrt(np.sum(
                    np.power(user_pos.reshape(user_pos.shape[0], 1, 2) -
                            router_coords.reshape(1, router_coords.shape[0], 2), 2),
                    axis=2))

                # Convertir distancias a latencias
                latencies = c * distances

                # Aplicar penalizaciones de manera vectorizada
                latencies = np.where(latencies > limites_superiores[i],
                                    latencies + penalidades[i],
                                    latencies)

                # Encontrar router más cercano para cada usuario
                router_cercano = np.argmin(latencies, axis=1)

                # Cálculo de latencia optimizado
                if i == 0:
                    # Para tipo1, usamos el máximo
                    L[i] = ponderacion[i] * np.max(np.min(latencies, axis=1)) / 100
                else:
                    # Para los demás tipos, suma de latencias mínimas
                    L[i] = ponderacion[i] * np.sum(latencies[np.arange(len(router_cercano)), router_cercano]) / 100

                # Actualizar capacidad de routers
                for router in range(self.router):
                    usuarios_en_router = np.where(router_cercano == router)[0]
                    capacidad_tipo[router] -= np.sum(usuarios[usuarios_en_router, 2])

            # Penalización por capacidad excedida
            penalizacion_capacidad = 10 if np.any(capacidad_tipo + row[:self.router] < 0) else 0
            L_total[k] = np.sum(L) + penalizacion_capacidad

        return L_total

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
        eval = self.fx_multiple_optimized(pop_tmp=pop_gen)
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
        df_result = {'generacion': np.arange(self.generations), 'optimal': [], 'avg': [], "dominio":[]}
        col_position = np.where(np.arange(self.router*3) % 3 != 2)[0]
        #routers_optimos = dominio[-1][col_position].reshape(self.router,2)
        # Inicialización de matrices de resultados.
        imagen = np.zeros(self.generations)
        pop_avg = np.zeros(self.generations)
        dominio = np.zeros((self.generations, 3*self.router))
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
            print(f"{generation} \t | {imagen[generation]} \t | {dominio[generation][col_position]}")

        dominio_str = [str(d) for d in dominio]
        # Almacenar resultados finales.
        df_result['optimal'] = imagen
        #df_result['optimal_tipo1'] = imagen_tipo1
        df_result['avg'] = pop_avg
        df_result["dominio"] = dominio_str
        df_result = pd.DataFrame(df_result)

        # Generar gráficos de evolución.
        self.plot_routers(df_result, dominio)
        #self.plot_optimal(df_result, dominio)

        # Guardar resultados en un archivo CSV.
        df_result.to_csv(f"./save_csv/resultado_{self.generations}.csv", index=False)


        return {'dominio': dominio, 'imagen': imagen}  # Retorna las mejores soluciones encontradas.

    def plot_routers(self, df_result, dominio,save_f=True):
        col_position = np.where(np.arange(self.router*3) % 3 != 2)[0]
        routers_optimos = dominio[-1][col_position].reshape(self.router,2)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        users = self.distribution_people

        ax.plot(users["tipo3"][:, 1], users["tipo3"][:, 0], ".", label="Priority 3", color="black")
        ax.plot(users["tipo2"][:, 1], users["tipo2"][:, 0], ".", label="Priority 2", color="purple")
        ax.plot(users["tipo1"][:, 1], users["tipo1"][:, 0], ".", label="Priority 1", color="lime")

        # Para los routers óptimos
        ax.plot(routers_optimos[:, 1], routers_optimos[:, 0], ".", color="red", markersize=15)
        if save_f:
            plt.savefig("./save_images/result_{}_{}.png".format(self.generations,self.router))
        else:
            plt.show()
        # i=0
        # for router in col_position:
        #     print(dominio[-1][i:router])
        #     ax.plot(dominio[-1][i:router], ".",c="red", markersize=20)
        #     i=router+1
        # plt.show()





    def plot_optimal(self, df_result, dominio):

        """
        Genera y guarda gráficos que muestran la evolución de la latencia durante las generaciones
        y la distribución óptima de los routers con respecto a los usuarios.

        Parámetros:
        - df_result: DataFrame con los valores de latencia óptima por generación.
        - dominio: Matriz con la mejor posición de routers en cada generación.
        """

        # Crear una figura con dos subgráficos
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Gráfico de latencia óptima general
        ax.plot(df_result.optimal.values)
        ax.set_title("Best individuo")  # Título del gráfico
        ax.set_xlabel("Generation")  # Etiqueta del eje X
        ax.set_ylabel("Latency (ms)")  # Etiqueta del eje Y

        # Gráfico de latencia para usuarios de prioridad 1
        # ax[1].plot(df_result.optimal_tipo1.values)
        # ax[1].set_title("Best latency on Priority 1")  # Título del gráfico
        # ax[1].set_xlabel("Generation")  # Etiqueta del eje X
        # ax[1].set_ylabel("Latency (ms)")  # Etiqueta del eje Y

        # Ajustar diseño de la figura y guardar la imagen
        fig.tight_layout()
        plt.savefig("optimal.jpg")

        # Obtener la distribución de usuarios
        users = self.init_people()

        # Generar imágenes de la distribución óptima del router en cada generación
        for i in range(self.generations):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            # Posición óptima del router en la generación actual
            ax.plot(dominio[i][0], dominio[i][1], "*", c="red", label="Router", markersize=30)
            # Posiciones de los usuarios por prioridad
            ax.plot(users["tipo1"][:, 1], users["tipo1"][:, 0], ".", label="Priority 1", c="lime", marker=".", edgecolor="black")
            ax.plot(users["tipo2"][:, 1], users["tipo2"][:, 0], ".", label="Priority 2", c="purple", marker=".", edgecolor="black")
            ax.plot(users["tipo3"][:, 1], users["tipo3"][:, 0], ".", label="Priority 3", c="black", marker=".", edgecolor="black")

            # Etiquetas de los ejes y título del gráfico
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Optimal Distribution of a Router")

            # Guardar la imagen de la distribución óptima en la carpeta para el GIF
            plt.savefig(f"./images_gif/generation_{i}.jpg")


    def create_gif(image_folder:str, gif_filename:str, duration:int=100):
        """
        Convierte imágenes JPG de una carpeta en un GIF animado que muestra la evolución de la distribución óptima.
        Parámetros:
        - image_folder: Ruta de la carpeta con las imágenes generadas.
        - gif_filename: Nombre del archivo GIF de salida.
        - duration: Tiempo de visualización de cada imagen en milisegundos (por defecto 100 ms por imagen).
        """

        # Obtener lista de archivos de imagen en la carpeta, ordenados alfabéticamente
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        images.sort()  # Ordenar las imágenes para mantener una secuencia cronológica

        # Lista para almacenar los frames del GIF
        frames = []
        # Abrir y procesar cada imagen de la secuencia
        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            img = Image.open(image_path)

            # Convertir a modo RGB si la imagen tiene otro formato
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frames.append(img)

        # Guardar el GIF con las imágenes procesadas
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0  # 0 significa que el GIF se reproducirá en bucle infinito

        )
        print(f"GIF creado exitosamente: {gif_filename}")

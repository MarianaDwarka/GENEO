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

class GAtelco:
    """
    Clase que implementa un Algoritmo Genético para optimizar la ubicación de routers
    en función de la latencia para diferentes tipos de usuarios.
    """
    num_routers = 1  # Número de routers a optimizar
    dimension=3
    def __init__(self, mu=0.75, eta=0.5, generations: int = 1000,
                 people_priority: dict = {"tipo1": 5000, "tipo2": 15000, "tipo3": 100000},
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
        self.num_user_p1 = people_priority["tipo1"]
        self.num_user_p2 = people_priority["tipo2"]
        self.num_user_p3 = people_priority["tipo3"]
        self.router = router
        self.mask = np.where(np.arange(self.router*3) % 3 != 2)[0]

    def population(self):
        """
        Genera una población inicial aleatoria de porcentajes de capacidad.
        """
        porcentaje_capacidad = np.random.uniform(low=0, high=100, size=(self.pop_size, self.router))
        return porcentaje_capacidad

    def mutation(self, pop_tmp, ind_optimo, index_optimo):
        """
        Aplica mutación a algunos individuos de la población.
        """
        eta_mat = np.random.random(self.pop_size)
        mut_eta = np.where(eta_mat <= self.eta)[0]
        count_mod = mut_eta.size ## esta variable determina cuales individuos se van a mutar
        pop_tmp[mut_eta, :] = np.random.uniform(low=0, high=100, size=(count_mod, self.router))
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

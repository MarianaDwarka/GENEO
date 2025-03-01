import pandas as pd
from ga.ga_phase1 import GAtelco
from ga.ga_dynamico import GAdynamic, usuarios_dataset
from models.model_phase1 import  prompt_interpretacion_planeacion, consult_db

generaciones = 2
upfs = 20

planeacion = GAtelco(generations=generaciones,
                    router=upfs,mu=0.8,eta=0.35).GA()

individuo_opt = planeacion["dominio"][-1]
n_routers = individuo_opt.size//3
matrix_upf_opt = individuo_opt.reshape(n_routers, 3)
ubicaciones_upf = matrix_upf_opt[:,:2].flatten()
capacidad_upf = matrix_upf_opt[:,2]
latencia_optima = planeacion["imagen"][-1]
response_dict = {"hora":[],"response":[]}

for i in range(1,24):
    print(f"Hora {i}")
    response_dict["hora"].append(i)
    ajuste_dinamico = GAdynamic(upf_planeacion=individuo_opt,
                                dataframe_hour=usuarios_dataset(f"./csvs/Hora_0{i}_MEX_v2.csv" if i<10 else f"./csvs/Hora_{i}_MEX_v2.csv"),
                                router=n_routers,mu=0.8,eta=0.4).GA()
    carga_dinamica_optima = ajuste_dinamico["dominio"][-1]
    response_dict["response"].append(consult_db(prompt_interpretacion_planeacion,
                                    pos=ubicaciones_upf,
                                    bw_total=capacidad_upf,
                                    carga_optima=carga_dinamica_optima))


pd.DataFrame(response_dict).to_csv("./save_csv/validacion_generaciones_{}_routers_{}.csv".format(generaciones,upfs), index=False)

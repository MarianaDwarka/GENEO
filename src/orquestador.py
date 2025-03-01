from ga.ga_phase1 import GAtelco
from ga.ga_dynamico import GAdynamic, usuarios_dataset
from models.model_phase1 import  prompt_interpretacion_planeacion, consult_db


planeacion = GAtelco(generations=10,
                    router=6,mu=0.8,eta=0.35).GA()

individuo_opt = planeacion["dominio"][-1]
n_routers = individuo_opt.size//3
matrix_upf_opt = individuo_opt.reshape(n_routers, 3)
ubicaciones_upf = matrix_upf_opt[:,:2].flatten()
capacidad_upf = matrix_upf_opt[:,2]
latencia_optima = planeacion["imagen"][-1]

print(consult_db(prompt_interpretacion_planeacion,
            pos=ubicaciones_upf,
            latency=latencia_optima,
            bw_total=capacidad_upf))
exit()
for i in range(24):
    print(f"Hora {i}")
    ajuste_dinamico = GAdynamic(upf_planeacion=individuo_opt,
                                dataframe_hour=usuarios_dataset(f"./csvs/Hora_0{i}_MEX_v2.csv" if i<10 else f"./csvs/Hora_{i}_MEX_v2.csv"),
                                router=n_routers,mu=0.8,eta=0.4).GA()

# prompt_db_vector.format_map({"total_bandwidth":bw_total, "p1_num":p1_num, "p2_num":p2_num, "p3_num":p3_num})

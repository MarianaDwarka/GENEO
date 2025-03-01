from ga_phase1 import GAtelco
from ga_dynamico import GAdynamic, usuarios_dataset

planeacion = GAtelco(generations=1000,
                    router=15,mu=0.8,eta=0.35).GA()

individuo_opt = planeacion["dominio"][-1]

ajuste_dinamico = GAdynamic(upf_planeacion=individuo_opt,
                            dataframe_hour=usuarios_dataset(),
                            router=individuo_opt.size//3,mu=0.8,eta=0.4).GA()

# prompt_db_vector.format_map({"total_bandwidth":bw_total, "p1_num":p1_num, "p2_num":p2_num, "p3_num":p3_num})

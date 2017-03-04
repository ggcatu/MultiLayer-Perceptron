import pandas
import numpy as np
import matplotlib.pyplot as plt


def leer_input(fichero):
    resultado = pandas.read_table(fichero, sep=" ", header = None)
    return resultado

def generar_df(in_matrix):
    in_matrix.columns = ["x", "y", "resultado"]
    return in_matrix

def normalizar(data_frame):
    cols = ["x","y"]
    data_frame[cols] = data_frame[cols].apply(lambda x: (x - x.mean())/ ( x.std()))
    return data_frame
    
tabla = generar_df(leer_input("datos_P2_EM2017_N1000.txt"))
tabla = normalizar(tabla)

import pandas
import numpy as np
import sys
import matplotlib.pyplot as plt
from red import Red

def leer_input(fichero):
	resultado = pandas.read_table(fichero,sep= " ", header = None)
	return resultado

def generar_df(in_matrix):
	in_matrix.columns = ["x","y","resultado"]
	in_matrix.insert(0,"bias",1)
	return in_matrix

def normalizar(data_frame):

	cols_to_norm = ["x","y"]
	data_frame[cols_to_norm] = data_frame[cols_to_norm].apply(lambda x: (x-x.mean())/x.std() )


prueba = generar_df(leer_input("Datos/datos_P2_EM2017_N500.txt"))
capas = 3
eta = 0.01
capa = []
for i in range(int(capas)):
	capa += [int(input("Introduzca la cantidad de neuronas de la capa " + str(i+1) + ": "))]
red = Red(capa,2,float(eta))

columnas = []
for columna in prueba:
	columnas.append(columna)

epoca = 0
error = 0
errorAnt = 1000
while (epoca < 5 and abs(errorAnt - error) >= 0.000001):
	epoca += 1
	print(epoca)
	errorAnt = error
	error = 0
	for index, row in prueba.iterrows():
		estimulo = []
		for j in range(len(columnas)-1):
			estimulo.append(row[columnas[j]])
		result = red.propagar(estimulo,row[columnas[-1]])
		error += (row[columnas[-1]] - max(result))**2
	error = error/(2*len(prueba))
	print(error)

buenos = 0
result = []

error_prueba = 0
falsos_positivos = 0
falsos_negativos = 0
fig, ax = plt.subplots()
circle2 = plt.Circle((10, 10), 6, color='b', fill=False)
ax.add_artist(circle2)

plot_helper = [0,0,0,0]
for i in range(len(prueba)):
	estimulo = [prueba["bias"][i],prueba["x"][i],prueba["y"][i]]
	result = red.calcular(estimulo)
	error_prueba += (prueba["resultado"][i] - max(result))**2
	if(max(result) >= 0.5):
		salida = 1
	else:
		salida = 0
	if(salida == prueba["resultado"][i]):
		if (salida == 0):
			lr = None if plot_helper[0] else "Holi" 
			plt.plot(prueba["x"][i], prueba["y"][i], 'ro', label=lr)
			plot_helper[0]+=1
		else:
			lr = None if plot_helper[1] else "Holi" 
			plt.plot(prueba["x"][i], prueba["y"][i], 'bo')
			plot_helper[1]+=1
	else:
		if (salida == 0):
			lr = None if plot_helper[2] else "Holi" 
			falsos_negativos += 1
			plt.plot(prueba["x"][i], prueba["y"][i], 'r*')
			plot_helper[2]+=1
		else:
			lr = None if plot_helper[3] else "Holi" 
			falsos_positivos += 1
			plt.plot(prueba["x"][i], prueba["y"][i], 'b*')
			plot_helper[3]+=1
error_prueba = error_prueba/(2*len(prueba))
print("Error en el Entrenamiento: " + str(error))
print("Error en la Prueba: " + str(error_prueba))
print("Falsos Positivos: " + str(falsos_positivos))
print("Falsos Negativos: " + str(falsos_negativos))
plt.axis([0, 20, 0, 20])
plt.axis('equal')
plt.legend(numpoints=1)
plt.show()
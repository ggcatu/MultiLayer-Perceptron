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
	return in_matrix

def normalizar(data_frame):
	cols_to_norm = ["x","y"]
	data_frame[cols_to_norm] = data_frame[cols_to_norm].apply(lambda x: (x-x.mean())/x.std() )

nombre = sys.argv[1]
capas = sys.argv[2]
eta = sys.argv[3]
pesos = sys.argv[4]
test_set = sys.argv[5] 
tabla = generar_df(leer_input(nombre))
prueba = generar_df(leer_input(test_set))

capa = []
for i in range(int(capas)):
	capa += [int(input("Introduzca la cantidad de neuronas de la capa " + str(i+1) + ": "))]
red = Red(capa,int(pesos),float(eta))

epoca = 0
error = 0
errorAnt = 1000
while ((epoca < 500) and (abs(errorAnt - error) >= 0.00001)):
	epoca += 1
	print(epoca)
	errorAnt = error
	error = 0
	for i in range(len(tabla)):
		estimulo = [tabla["x"][i],tabla["y"][i]]
		result = red.propagar(estimulo,tabla["resultado"][i])
		error += (tabla["resultado"][i] - result)**2
	error = error/(2*len(tabla))
	print(error)
error_prueba = 0
falsos_positivos = 0
falsos_negativos = 0
fig, ax = plt.subplots()
circle2 = plt.Circle((10, 10), 6, color='b', fill=False)
ax.add_artist(circle2)
for i in range(len(prueba)):
	estimulo = [prueba["x"][i],prueba["y"][i]]
	result = red.calcular(estimulo)
	error_prueba += (prueba["resultado"][i] - result)**2
	if(result[0] >= 0.5):
		salida = 1
	else:
		salida = 0
	if(salida == prueba["resultado"][i]):
		if (salida == 0):
			plt.plot(prueba["x"][i], prueba["y"][i], 'ro')
		else:
			plt.plot(prueba["x"][i], prueba["y"][i], 'bo')
	else:
		if (salida == 0):
			falsos_negativos += 1
			plt.plot(prueba["x"][i], prueba["y"][i], 'r*')
		else:
			falsos_positivos += 1
			plt.plot(prueba["x"][i], prueba["y"][i], 'b*')
error_prueba = error_prueba/(2*len(prueba))
print("Error en el Entrenamiento: " + str(error[0]))
print("Error en la Prueba: " + str(error_prueba[0]))
print("Falsos Positivos: " + str(falsos_positivos))
print("Falsos Negativos: " + str(falsos_negativos))
plt.axis([0, 20, 0, 20])
plt.axis('equal')
plt.show()
	
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
def leer_input_iris(fichero):
	resultado = pandas.read_table(fichero,sep= ",", header = None)
	return resultado

def generar_df_iris(in_matrix):
	in_matrix.columns = ["x","y","z","w","resultado"]
	return in_matrix

def normalizar_iris(data_frame):
	cols_to_norm = ["x","y","z","w"]
	data_frame[cols_to_norm] = data_frame[cols_to_norm].apply(lambda x: (x-x.mean())/x.std() )

def binario_iris(data_frame):
	data_frame["resultado"] = data_frame["resultado"].apply(lambda x: int(x == "Iris-setosa"))

def iris(data_frame):
	for i in range(len(data_frame)):
		if(data_frame["resultado"][i] == "Iris-setosa"):
			data_frame["resultado"][i] = 0
		elif(data_frame["resultado"][i] == "Iris-versicolor"):
			data_frame["resultado"][i] = 1
		else:
			data_frame["resultado"][i] = 2

if(len(sys.argv) != 7):
	print("Uso: python3 " + sys.argv[0] + " problema tipo instancia capas learningRate test")
	print("")
	print("   problema: 0 para patrones (x,y), 1 para datos iris")
	print("   tipo: 0 para clasificacion binaria, 1 para clasificacion por clases")
	print("   instancia: Ruta del archivo con las instancias")
	print("   capas: Cantidad de capas que tendra la red")
	print("   learningRate: Factor de aprendizaje para la red")
	print("   test: En caso de ser el problema 0 la ruta del archivo de validacion") 
	print("         Si es el problema 1 el porcentaje de datos para entranar")
	sys.exit()


prueba = generar_df(leer_input(test_set))

capa = []
for i in range(int(capas)):
	capa += [int(input("Introduzca la cantidad de neuronas de la capa " + str(i+1) + ": "))]
red = Red(capa,2,float(eta))

epoca = 0
error = 0
errorAnt = 1000
while (epoca < 500 and abs(errorAnt - error) >= 0.000001):
	epoca += 1
	print(epoca)
	errorAnt = error
	error = 0
	for index, row in tabla.iterrows():
		estimulo = []
		for j in range(len(columnas)-1):
			estimulo.append(row[columnas[j]])
		result = red.propagar(estimulo,row[columnas[-1]])
		error += (row[columnas[-1]] - max(result))**2
	error = error/(2*len(tabla))
	print(error)

buenos = 0
result = []

error_prueba = 0
falsos_positivos = 0
falsos_negativos = 0
fig, ax = plt.subplots()
circle2 = plt.Circle((10, 10), 6, color='b', fill=False)
ax.add_artist(circle2)
for i in range(len(prueba)):
	estimulo = [prueba["x"][i],prueba["y"][i]]
	result = red.calcular(estimulo)
	error_prueba += (prueba["resultado"][i] - max(result))**2
	if(max(result) >= 0.5):
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
print("Error en el Entrenamiento: " + str(error))
print("Error en la Prueba: " + str(error_prueba))
print("Falsos Positivos: " + str(falsos_positivos))
print("Falsos Negativos: " + str(falsos_negativos))
plt.axis([0, 20, 0, 20])
plt.axis('equal')
plt.show()
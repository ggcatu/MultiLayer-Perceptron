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
	print("   test: Ruta del archivo para validar")
	sys.exit()


problema = sys.argv[1]
tipo = sys.argv[2]
nombre = sys.argv[3]
capas = sys.argv[4]
eta = sys.argv[5]
test_set = sys.argv[6]

if(problema == "0"):
	prueba = generar_df(leer_input(test_set))
	tabla = generar_df(leer_input(nombre))
	#normalizar(tabla)
else:
	tabla = generar_df_iris(leer_input_iris(nombre))
	if(tipo == "0"):
		binario_iris(tabla)
	else:
		iris(tabla)
	#normalizar_iris(tabla)

columnas = []
for columna in tabla:
	columnas.append(columna)

capa = []
for i in range(int(capas)):
	capa += [int(input("Introduzca la cantidad de neuronas de la capa " + str(i+1) + ": "))]
red = Red(capa,len(columnas)-1,float(eta))

epoca = 0
error = 0
errorAnt = 1000
while (epoca < 500 and abs(errorAnt - error) >= 0.000001):
	epoca += 1
	print(epoca)
	errorAnt = error
	error = 0
	for i in range(len(tabla)):
		estimulo = []
		for j in range(len(columnas)-1):
			estimulo.append(tabla[columnas[j]][i])
		result = red.propagar(estimulo,tabla[columnas[-1]][i])
		error += (tabla[columnas[-1]][i] - max(result))**2
	error = error/(2*len(tabla))
	print(error)

buenos = 0
result = []

fig, ax = plt.subplots()
circle2 = plt.Circle((10, 10), 6, color='b', fill=False)
ax.add_artist(circle2)

'''
for i in range(len(tabla)):
	estimulo = []
	for j in range(len(columnas)-1):
		estimulo.append(tabla[columnas[j]][i])
	result = red.calcular(estimulo)
	print(result)
	if(max(result) < 1):
		salida = 0
	elif(max(result) >= 1 and max(result) < 2):
		salida = 1
	else:
		salida = 2
	if(salida == tabla["resultado"][i]):
		buenos += 1
print(buenos)

'''
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
print("Error en el Entrenamiento: " + str(error))
print("Error en la Prueba: " + str(error_prueba))
print("Falsos Positivos: " + str(falsos_positivos))
print("Falsos Negativos: " + str(falsos_negativos))
plt.axis([0, 20, 0, 20])
plt.axis('equal')
plt.show()

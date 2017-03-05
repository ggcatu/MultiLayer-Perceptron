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
tabla = generar_df(leer_input(nombre))

capa = []
for i in range(int(capas)):
	capa += [int(input("Introduzca la cantidad de neuronas de la capa " + str(i+1) + ": "))]
red = Red(capa,int(pesos),float(eta))

epoca = 0
error = 0
errorAnt = 1000
while (epoca < 500 and abs(errorAnt - error) >= 0.00001):
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

buenos = 0

fig, ax = plt.subplots()
circle2 = plt.Circle((10, 10), 6, color='b', fill=False)
ax.add_artist(circle2)
for i in range(len(tabla)):
	estimulo = [tabla["x"][i],tabla["y"][i]]
	result = red.calcular(estimulo)
	if(result[0] >= 0.5):
		salida = 1
	else:
		salida = 0

	if(salida == tabla["resultado"][i]):
		if (salida == 0):
			plt.plot(tabla["x"][i], tabla["y"][i], 'ro')
		else:
			plt.plot(tabla["x"][i], tabla["y"][i], 'bo')
		buenos += 1
	else:
		if (salida == 0):
			plt.plot(tabla["x"][i], tabla["y"][i], 'r*')
		else:
			plt.plot(tabla["x"][i], tabla["y"][i], 'b*')
print(str(buenos) + " " + str(500))
plt.axis([0, 20, 0, 20])
plt.axis('equal')
plt.show()
	
import numpy
import sys
import pandas
from main import Neuron

class Red():

	def __init__(self,arregloN, atr):
		self.red = []
		numero = atr
		eta = 0.1
		for i in range(0,len(arregloN)):
			if(i != len(arregloN)-1):
				self.red += [[Neuron(eta,numero) for j in range(arregloN[i])]]
			else:
				self.red += [[Neuron(eta,numero,lineal = True,salida = True) for j in range(arregloN[i])]]
			numero = arregloN[i]

	def __calc(self, entradas):
		estimulos = [entradas]
		for i in range(len(self.red)):
			estimulos += [[]]
			for j in range(len(self.red[i])):
				estimulo = self.red[i][j].calcular(estimulos[i])
				estimulos[i+1].append(estimulo)
		return estimulos

	def calcular(self, entradas):
		return self.__calc(entradas)[-1]

	# Paso primero las entradas y propago hacia adelante
	# haciendo un arreglo de estimulos en el cual esta que
	# le entra a la capa [i]
	def propagar(self,entradas,resultado):
		estimulos = self.__calc(entradas)

		# Tecnicamente luego de propagar hacia adelante, hago
		# el backprop de cada capa.
		i = len(self.red)-1
		k = len(estimulos)-2
		gradientes = []
		while(i >= 0):
			gradientes = self.backprop(i,estimulos[k], gradientes,resultado)
			k -= 1
			i -= 1
		return estimulos[-1]

	# Propago una capa hacia atras, la capa i 
	# la recorro, calculo sus gradientes y actualizo con
	# las reglas para cada capa y retorno los gradientes
	def backprop(self,i,estimulos,gradientes,resultado):
		grad =[]
		if(i == len(self.red)-1):
			for j in range(len(self.red[i])):
				salida = self.red[i][j].calcular(estimulos)
				gradiente = self.red[i][j].gradienteLocal(resultado-salida)
				grad += [gradiente]
				ve = [gradiente*estimulos[k]*self.red[i][j].eta for k in range(len(estimulos))]
				self.red[i][j].actualizarPesos(ve)
		else:
			for j in range(len(self.red[i])):
				error = sum(gradientes[k]*self.red[i+1][k].pesos[j] for k in range(len(gradientes)))
				gradiente = self.red[i][j].gradienteLocal(error)
				grad += [gradiente]
				ve = [gradiente*estimulos[k]*self.red[i][j].eta for k in range(len(estimulos))]
				self.red[i][j].actualizarPesos(ve)	
		return grad

if __name__ == "__main__":
	red = Red([2,3,1], 2)
	print(red.red[2][0].pesos)
	print(red.calcular([2,3]))
	print(red.red[2][0].pesos)
	print(red.propagar([2,3],5))
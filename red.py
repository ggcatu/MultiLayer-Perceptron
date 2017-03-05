import numpy
import sys
import pandas
from main import Neuron

class Red():

	def __init__(self,arregloN, atr, resultados):
		self.red = []
		self.resultado = resultados
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
		return self.__calc(entradas)[-1][0]
	# Paso primero las entradas y propago hacia adelante
	# haciendo un arreglo de estimulos en el cual esta que
	# le entra a la capa [i]
	def propagar(self,entradas):
		estimulos = self.__calc(entradas)

		# Tecnicamente luego de propagar hacia adelante, hago
		# el backprop de cada capa.
		i = len(self.red)-1
		k = len(estimulos)-1
		gradientes = []
		while(i >= 0):
			gradientes = backprop(i,estimulos[k], gradientes)
			k -= 1
			i -= 1

	# Propago una capa hacia atras, la capa i 
	# la recorro, calculo sus gradientes y actualizo con
	# las reglas para cada capa y retorno los gradientes
	def backprop(self,i,estimulos,gradientes):
		grad =[]
		if(i == len(self.red)-1):
			for j in range(len(self.red[i])):
				salida = self.red[i][j].calcular(entradas)
				gradiente = self.red[i][j].gradienteLocal(self.resultado-salida)
				grad += [gradiente]
				self.red[i][j].actualizarPesos
				([gradiente*estimulos[k]*self.red[i][j].eta for k in range(len(estimulos))])
		else:
			for j in range(len(self.red[i])):
				gradiente = self.red[i][j].gradienteLocal
				(sum(gradientes[k]*estimulos[k] for k in range(len(gradientes))))
				grad += [gradiente]
				self.red[i][j].actualizarPesos
				([gradiente*estimulos[k]*self.red[i][j].eta for k in range(len(estimulos))])	
		return grad

if __name__ == "__main__":
	red = Red([2,3,1], 2, 5)
	print(red.calcular([2,3]))
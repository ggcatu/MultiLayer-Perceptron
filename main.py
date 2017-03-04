import numpy

class Neuron():
    '''
        Eta: Tasa de aprendizaje de la neurona
        Pesos: Vector de pesos iniciales
        Lineal: Indica si la neurona es lineal (True) o sigmoidal (False por defecto)
        Salida: Indica si eres una neurona de salida, pues el gradiente es diferente
    '''
    def __init__(self,eta = 0.01, pesos = [], lineal = False, salida = False):
        self.pesos = pesos
        self.lineal = lineal
        self.eta = eta
        self.salida = salida
        self.campo = 0
        self.funcion = lambda x: 1-math.exp(-x)
        self.dfuncion = lambda x: math.exp(-x)

    def __producto(self,entradas):
        '''
            Metodo privado, utilizar Calcular
            Entradas: Vector a evaluar en la neurona
        '''
        if len(self.pesos) != len(entradas):
            print("Error de entrada")
        return sum(entrada*peso for entrada,peso in zip(entradas,self.pesos))

    def calcular(self,entradas):
        '''
            Calcula el resultado de procesar entrada, en la neurona
            Entradas: Vector a evaluar en la neurona
        '''
        self.campo = self.__producto(entradas)
        self.entradas = entradas
        if self.lineal:
            #print("Lineal|| "+str(self.campo))
            return self.campo
        else:
            #print("Sigmoid|| "+str(self.funcion(self.campo)))
            return self.funcion(self.campo)

    def gradienteLocal(self,error):
        '''
            Calcula el gradiente local de la neurona
            error: Si es de salida, vector debe ser el error de la neurona Dj - Yj
                    Si es oculta, debe ser la sumatoria de los gradientes de las neuronas
                    posteriores, por el correspondiente peso entre las mismas
        '''
        if self.lineal:
            return error
        else:
            return error * self.dfuncion(self.campo)

    def actualizarPesos(self,vector):
        '''
            Actualiza los pesos de la neurona
            vector: Vector de pesos a sumar
        '''
        self.pesos = [ x + y for x, y in zip(self.pesos, vector)]
	

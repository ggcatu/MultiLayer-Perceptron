import random
import pandas

def punto():
    return [random.uniform(0, 20),random.uniform(0, 20)]

def esta_dentro(punto):
    x,y = punto
    if (x-10)**2 + (y-10)**2 < 36:
        return True
    return False
    
def generar():
    d = []
    for i in range(0,100):
        for j in range(0,100):
            punto = [random.uniform(0.2*j, 0.2*(j+1)),random.uniform(0.2*i, 0.2*(i+1))]
            d.append(punto + [int(esta_dentro(punto))])
    return d

d = generar()
tabla = pandas.DataFrame(data = d)
tabla.to_csv("set_prueba_N10000.txt",sep = " ", header=False, index = False)
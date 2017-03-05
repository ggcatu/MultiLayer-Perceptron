import random
import pandas

def punto():
    return [random.uniform(0, 20),random.uniform(0, 20)]

def esta_dentro(punto):
    x,y = punto
    if (x-10)**2 + (y-10)**2 < 36:
        return True
    return False
    
def generar(numero):
    c_d = numero / 2
    c_f = c_d
    d = []
    f = []
    while len(d)<c_d and len(f) < c_f:
        g = punto()
        r = esta_dentro(g)
        if (r):
            d.append(g + [int(r)])
        else :
            f.append(g + [int(r)])
    if len(d)<c_d:
        d = completar(d,c_d,True)
    else:
        f = completar(f,c_f,False)
    d = d + f
    random.shuffle(d)
    return d

def completar(d,limite,boo):
    while len(d) < limite:
        g = punto()
        if (esta_dentro(g) == boo):
            d.append(g+[int(boo)])
    return d

d = generar(2000)
tabla = pandas.DataFrame(data = d)
tabla.to_csv("gen_N2000.txt",sep = " ", header=False, index = False)
        



    



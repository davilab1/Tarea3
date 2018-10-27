import numpy as np
import matplotlib.pylab as plt
import math
import scipy as sp
from scipy.interpolate import interp1d


#Almacene los datos de signal.dat y de incompletos.dat
datosignal=np.genfromtxt("signal.dat")
datosincom=np.genfromtxt("incompletos.dat")
#print(datosignal)
#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla en ApellidoNombre_signal.pdf.

plt.figure()
plt.plot(datosignal,label="signal")
plt.title('Datos de signal.dat')
plt.legend(loc="best")
#plt.savefig("AvilaDario_signal.pdf")

'''
#Haga la transformada de Fourier de los datos de la senal usando su implementacion propia de la transformada discreta de fourier.


#Haga una grafica de la transformada de Fourier y guarde dicha grafica sin mostrarla en ApellidoNombre_TF.pdf.
plt.figure()
plt.plot()
plt.title('')
plt.show()
plt.xlabel()
plt.ylabel()
#plt.savefig("AvilaDario_TF.pdf")

#Esta grafica debe ser en funcion de las frecuencias (bono de 3pts si no usa el paquete fftfreq. Indique esto con un mensaje en la terminal.)


#Imprima un mensaje donde indique cuales son las frecuencias principales de su senal.


#Haga un filtro pasa bajos con frecuencia de corte fc = 1000Hz. realice la transformada inversa y
#haga una grafica de la senal filtrada. Guarde dicha grafica sin mostrarla en ApellidoNombre_filtrada.pdf.

fc1=1000
plt.figure()
plt.plot()
plt.title('')
plt.show()
plt.xlabel()
plt.ylabel()
#plt.savefig("AvilaDario_filtrada.pdf")

#scriba un mensaje en la terminal explicando por que no puede hacer la transformada de Fourier de los datos de incompletos.dat
#print ("No se puede hacer la transformada de fourier utilizando los datos de datosincompletos.dat ya que ")

#Haga una interpolacion cuadratica y una cubica de sus datos incompletos.dat con 512 puntos.
puntos=512
def interlineal(xx,yy):
    ilineal=interp1d(xx,yy,kind='linear')
    return ilineal

def intercuadrat(xx,yy):
    icuadrat=interp1d(xx,yy,kind='quadratic')
    return icuadrat

interplineal=interlineal()
interpcuad=intercuadrat()

# Haga la trasformada de Fourier de cada una de las series de datos interpoladas.


# Haga unag rafica con tres subplots delas tres transformada deFourier(datosdesignal.dat y datos interpolados) y guardela sin mostrarla en ApellidoNombre_TF_interpola.pdf.
'''

plt.figure()
plt.subplot(211)
plt.title('Tranformada de Datos')
plt.plot()
plt.subplot(212)
plt.title('Transformada de interpolacion lineal')
plt.plot()
plt.subplot(213)
plt.title('Transformada de interpolacion cuadratica')
plt.plot()
#plt.savefig("AvilaDario_TF_interpola.pdf")



#Imprima un mensaje donde describa las diferencias encontradas entre la transformada de Fourier de la senal original y las de las interpolaciones.
print("Las diferencias encontradas entre la transformada de Fourier de la signal original y las interpolaciones es que basicamente... ")

# Aplique el filtro pasabajos con una frecuencia de corte fc = 1000Hz y con una frecuencia de corte de fc = 500Hz.
fc2=500

# Haga una grafica con dos subplots (uno para cada filtro) de las 3 senales filtradas y guardela sin mostrarla en ApellidoNombre_2Filtros.pdf.

plt.figure()
plt.plot()
plt.title('')
plt.show()
plt.xlabel()
plt.ylabel()
#plt.savefig("AvilaDario_2Filtros.pdf")

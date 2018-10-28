import numpy as np
import matplotlib.pylab as plt
import math
from scipy.fftpack import fft, fftfreq
import scipy as sp
from scipy.interpolate import interp1d

#Almacene los datos de signal.dat y de incompletos.dat
datosignal=np.genfromtxt("signal.dat",delimiter=",")
datosincom=np.genfromtxt("incompletos.dat",delimiter=",")

datxsignal=datosignal[:,0]
datysignal=datosignal[:,1]
datxinc=datosincom[:,0]
datyinc=datosincom[:,1]
#print(datosxsignal)
#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla en ApellidoNombre_signal.pdf.
n=len(datosignal)
n1=len(datosincom)
print (n)
print (n1)

plt.figure()
plt.plot(datxsignal,datysignal,label="signal")
plt.title('Datos de signal.dat')
plt.legend(loc="best")
#plt.savefig("AvilaDario_signal.pdf")


#Haga la transformada de Fourier de los datos de la senal usando su implementacion propia de la transformada discreta de fourier.
def transFourier(N,datos):

    transformada=np.zeros((n,),dtype=np.complex)
    for i in range(N):
        for j in range(N):
            transformada[i]+=datos[j]*np.exp(-2j*np.pi*j*i/N)
    return transformada

signalTransf=transFourier(n,datysignal)
#Haga una grafica de la transformada de Fourier y guarde dicha grafica sin mostrarla en ApellidoNombre_TF.pdf.

plt.figure()
plt.plot(signalTransf)
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
#plt.show()
#plt.savefig("AvilaDario_TF.pdf")

#Esta grafica debe ser en funcion de las frecuencias (bono de 3pts si no usa el paquete fftfreq. Indique esto con un mensaje en la terminal.)
frecu=fftfreq(n)
#print(frecu)
#Imprima un mensaje donde indique cuales son las frecuencias principales de su senal.
print("Las frecuencias principales de la señal son aquellas cercanas a 0 y a 500Hz")

#Haga un filtro pasa bajos con frecuencia de corte fc = 1000Hz. realice la transformada inversa y
#haga una grafica de la senal filtrada. Guarde dicha grafica sin mostrarla en ApellidoNombre_filtrada.pdf.

def invTransFourier(N,transformada):

    invtransformada=np.zeros((n,),dtype=np.complex)
    for i in range(N):
        for j in range(N):
            invtransformada[i]+=transformada[j]*np.exp(2j*np.pi*j*i/n)

    return invtransformada


def pasarbajos(freq,ftran,filtro):
    for i in range(len(freq)):
        if(abs(freq[i]>filtro)):
            ftran[i]=0
        else:
            ftran[i]=ftran[i]
    return ftran

fc1=1000
fftbajos=pasarbajos(frecu,signalTransf,fc1)
#print (np.real(fftbajos))


plt.figure()
plt.plot(frecu,fftbajos)
plt.title('Tranformada Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
#plt.show()
#plt.savefig("AvilaDario_filtrada.pdf")

#scriba un mensaje en la terminal explicando por que no puede hacer la transformada de Fourier de los datos de incompletos.dat
print ("No se puede hacer la transformada de fourier utilizando los datos de datosincompletos.dat ya que en principio la diferencia de los datos con respecto a singal es la cantidad de datos que se tienen, un arreglo posee 512 elementos mientras que el incompleto presenta 117 ")

#Haga una interpolacion cuadratica y una cubica de sus datos incompletos.dat con 512 puntos.
'''puntos=512
def interlineal(xx,yy):
    ilineal=interp1d(xx,yy,kind='linear',fill_value=512)
    return ilineal

def intercuadrat(xx,yy):
    icuadrat=interp1d(xx,yy,kind='quadratic',fill_value=512)
    return icuadrat

interplineal=interlineal(frecu,datosincom)
interpcuad=intercuadrat(frecu,datosincom)'''

# Haga la trasformada de Fourier de cada una de las series de datos interpoladas.


# Haga unag rafica con tres subplots delas tres transformada deFourier(datosdesignal.dat y datos interpolados) y guardela sin mostrarla en ApellidoNombre_TF_interpola.pdf.


plt.figure()
plt.subplot(311)
plt.title('Tranformada de Datos')
plt.plot()
plt.subplot(312)
plt.title('Transformada de interpolacion lineal')
plt.plot()
plt.subplot(313)
plt.title('Transformada de interpolacion cuadratica')
plt.plot()
plt.tight_layout()

#plt.savefig("AvilaDario_TF_interpola.pdf")



#Imprima un mensaje donde describa las diferencias encontradas entre la transformada de Fourier de la senal original y las de las interpolaciones.
print("Las diferencias encontradas entre la transformada de Fourier de la signal original y las interpolaciones es que basicamente... ")

# Aplique el filtro pasabajos con una frecuencia de corte fc = 1000Hz y con una frecuencia de corte de fc = 500Hz.
fc2=500
#aplicando frecuencia de corte 1000 para las 3 signals
'''filtermildatos=pasarbajos()
#print (np.real())

filtermillineal=pasarbajos()
filtermilcuadrat=pasarbajos()




#aplicando frecuencia de corte 500 para las 3 signals
filterquindatos=pasarbajos()
#print (np.real())
filterquinlineal=pasarbajos()
filterquincuadrat=pasarbajos()'''

# Haga una grafica con dos subplots (uno para cada filtro) de las 3 senales filtradas y guardela sin mostrarla en ApellidoNombre_2Filtros.pdf.

plt.figure()

plt.subplot(321)
plt.plot()
plt.title('Filtro de 1000Hz para los datos')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.subplot(322)
plt.plot()
plt.title('Filtro de 1000Hz para los datos lineales')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.subplot(323)
plt.plot()
plt.title('Filtro de 1000Hz para los datos cuadraticos')
plt.xlabel('Frecuencia(Hz)')
plt.ylabel('Amplitud')

plt.subplot(324)
plt.plot()
plt.title('Filtro de 500Hz para los datos')
plt.xlabel('Frecuencia(Hz)')
plt.ylabel('Amplitud')

plt.subplot(325)
plt.plot()
plt.title('Filtro de 500Hz para los datos lineales')
plt.xlabel('Frecuencia(Hz)')
plt.ylabel('Amplitud')

plt.subplot(326)
plt.plot()
plt.title('Filtro de 500Hz para los datos cuadraticos')
plt.xlabel('Frecuencia(Hz)')
plt.ylabel('Amplitud')
plt.tight_layout()
#plt.savefig("AvilaDario_2Filtros.pdf")'''
print("pendejo descomentar los savefig!!!!!")

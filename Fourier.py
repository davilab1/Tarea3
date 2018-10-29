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


#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla en ApellidoNombre_signal.pdf.
n=len(datosignal)
n1=len(datosincom)
#print (n)#512
#print (n1)#117

espaciado=datxsignal[1]-datxsignal[0]
SamplR=1/espaciado

plt.figure()
plt.plot(datxsignal,datysignal,label="signal")
plt.title('Datos de signal.dat')
plt.legend(loc="best")
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.savefig("AvilaDario_signal.pdf")

#Haga la transformada de Fourier de los datos de la senal usando su implementacion propia de la transformada discreta de fourier.
def transFourier(N,datos):
    transformada=np.zeros((N,),dtype=np.complex)
    for i in range(N):
        for j in range(N):
            transformada[i]+=datos[j]*np.exp(-2j*np.pi*j*i/N)
    return transformada

#signalprueba=trans(datxsignal,datysignal,n)
signalTransf=transFourier(n,datysignal)
#print(len(signalTransf))
frecu=fftfreq(n,espaciado)
#Haga una grafica de la transformada de Fourier y guarde dicha grafica sin mostrarla en ApellidoNombre_TF.pdf.
plt.figure()
plt.plot(frecu,signalTransf,label="Transformada de Fourier para Signal")
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend(loc="best")
#plt.show()
#plt.savefig("AvilaDario_TF.pdf")

#Imprima un mensaje donde indique cuales son las frecuencias principales de su senal.
print("Las frecuencias principales de la señal son aquellas frecuencias que presenten una mayor amplitud. Si se observa el grafico de la tranformada de Fourier, es posible observar que hay un rango de frecuencias de -7500 a 7500 Hz y que en su mayoria, las frecuencias mas importantes se encuentran alrededor de los 1000Hz. Es por esto que todas las otras frecuencias generan un ruido, el cual debe ser filtrado.")

#Haga un filtro pasa bajos con frecuencia de corte fc = 1000Hz. realice la transformada inversa y
#haga una grafica de la senal filtrada. Guarde dicha grafica sin mostrarla en ApellidoNombre_filtrada.pdf.

def invTransFourier(N,transformada):

    invtransformada=np.zeros((N,),dtype=np.complex)
    for i in range(N):
        for j in range(N):
            invtransformada[i]+=transformada[j]*np.exp(2j*np.pi*j*i/N)

    return invtransformada


def pasarbajos(freq,ftran,filtro):
    for i in range(len(freq)):
        if(abs(freq[i]>filtro)):
            ftran[i]=0
        else:
            ftran[i]=ftran[i]
    return ftran

fc1=1000

#signalinvTransf=invTransFourier(n,signalTransf)
#invfftbajos=pasarbajos(frecu,signalinvTransf,fc1)
inversafiltrada=invTransFourier(n,pasarbajos(frecu,signalTransf,fc1))
#filtro2=pasarbajos(frecu,signalTransf,fc1)
#inv2=invTransFourier(n,filtro2)

#print (np.real(fftbajos))
plt.figure()
#plt.plot(datxsignal,datysignal,"o",label="Signal original",color="black")
plt.plot(datxsignal,np.real(inversafiltrada), label="Inversa de Signal Filtrada",color="red")
plt.title('Tranformada Inversa Filtrada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="best")

#plt.savefig("AvilaDario_filtrada.pdf")

#scriba un mensaje en la terminal explicando por que no puede hacer la transformada de Fourier de los datos de incompletos.dat
print ("En teoria si se le puede hacer la transformada de fourier utilizando los datos de datosincompletos.dat ya que en principio la diferencia de los datos con respecto a singal es la cantidad de datos que se tienen, un arreglo posee 512 elementos mientras que el incompleto presenta 117, asi que al tener una menor cantidad de frecuencias, presenta una menor cantidad de ruido. Por lo tanto, no seria tan util realizar la tranformada de fourier para esots datos  ")
from scipy.interpolate import UnivariateSpline #paquete necesario para poder ajustar los datos a 512 puntos antes de realizar la interpolacion

datxinc=datosincom[:,0]
datyinc=datosincom[:,1]

#Haga una interpolacion cuadratica y una cubica de sus datos incompletos.dat con 512 puntos.
nuevospuntos=512
incompletosViejos=np.arange(0,len(datosincom))
nuevosIncompletos=np.linspace(0,len(datosincom)-1,nuevospuntos)

splx=UnivariateSpline(incompletosViejos,datxinc)
nuevoArrIncompetox=splx(nuevosIncompletos)
#print(len(nuevoArrIncompetox))

sply=UnivariateSpline(incompletosViejos,datyinc)
nuevoArrIncompetoy=sply(nuevosIncompletos)


#print(len(nuvevoArrIncompeto))
def intercubic(xx,yy):
    icubica=interp1d(xx,yy,kind='cubic')
    return icubica

def intercuadrat(xx,yy):
    icuadrat=interp1d(xx,yy,kind='quadratic')
    return icuadrat
'''
def interpola(datosin,newdata):
    xx=np.linspace(min(datosin),max(datosin),512)
    intercubic=interp1d(datosin,newdata,kind='quadratic')
    intercuad=interp1d(datosin,newdata,kind='quadratic')
    intcubica=intercubic(newdata)
    intcuadrat=intercuad(newdata)

    return 0'''

interpcubica=intercubic(nuevoArrIncompetoy,nuevoArrIncompetox)
intercubicarr=interpcubica(frecu)

interpcuad=intercuadrat(nuevoArrIncompetoy,nuevoArrIncompetox)
intercuadarr=interpcuad(frecu)

# Haga la trasformada de Fourier de cada una de las series de datos interpoladas.
cubicTransf=transFourier(n,intercubicarr)
cuadratTransf=transFourier(n,intercuadarr)

# Haga unag rafica con tres subplots delas tres transformada deFourier(datosdesignal.dat y datos interpolados) y guardela sin mostrarla en ApellidoNombre_TF_interpola.pdf.
plt.figure()
plt.subplot(311)
plt.title('Tranformada de Datos')
plt.plot(frecu,signalTransf)
plt.subplot(312)
plt.title('Transformada de interpolacion cubica')
plt.plot(frecu,cubicTransf)
plt.subplot(313)
plt.title('Transformada de interpolacion cuadratica')
plt.plot(frecu,cuadratTransf)
plt.tight_layout()
plt.show()
#plt.savefig("AvilaDario_TF_interpola.pdf")'''

#Imprima un mensaje donde describa las diferencias encontradas entre la transformada de Fourier de la senal original y las de las interpolaciones.
print("Las diferencias encontradas entre la transformada de Fourier de la signal original y las interpolaciones es que basicamente... ")
# Aplique el filtro pasabajos con una frecuencia de corte fc = 1000Hz y con una frecuencia de corte de fc = 500Hz.
fc2=500

#aplicando frecuencia de corte 1000 para las 3 signals
filtermildatos=pasarbajos(frecu,signalTransf,fc1)
filtermilcubic=pasarbajos(frecu,cubicTransf,fc1)
filtermilcuadrat=pasarbajos(frecu,cuadratTransf,fc1)

#aplicando frecuencia de corte 500 para las 3 signals
filterquindatos=pasarbajos(frecu,signalTransf,fc2)
filterquincubic=pasarbajos(frecu,cubicTransf,fc2)
filterquincuadrat=pasarbajos(frecu,cuadratTransf,fc2)


# Haga una grafica con dos subplots (uno para cada filtro) de las 3 senales filtradas y guardela sin mostrarla en ApellidoNombre_2Filtros.pdf.

plt.figure()
plt.subplot(211)
plt.plot(frecu,filtermildatos,label='Signal',color="green")
plt.plot(frecu,filtermilcubic,label='Signal Cubica',color="blue")
plt.plot(frecu,filtermilcuadrat,label="Signal Cuadratica",color="red")
plt.title('Filtro de 1000Hz')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend(loc="best")


plt.subplot(212)
plt.plot(frecu, filterquindatos,label='Signal',color="green")
plt.title('Filtro de 500Hz')
plt.plot(frecu,filterquincubic,label='Signal Cubica',color="blue")
plt.plot(frecu,filterquincuadrat,label='Signal Cuadratica',color="red")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend(loc="best")

plt.tight_layout()
plt.show()
#plt.savefig("AvilaDario_2Filtros.pdf")'''
print("descomentar los savefig!!!!!")

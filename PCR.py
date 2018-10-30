import numpy as np
import matplotlib.pylab as plt
import math

#Almacene los datos del archivo WDBC.dat.
datos=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
columna=datos[:,0]
#print (len(columna)) 569
fila=datos[0,:]
#print(len(fila)) 30
#https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
info=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(1),dtype="unicode")
#print(len(info))
# Calcule, con su implementacion propia, la matriz de covarianza de los datos y la imprima

def pca2(dat,filas,columnas):

    mat=np.zeros((len(filas),len(filas)))

    for i in range(len(fila)):
        for j in range(len(fila)):

            x1=dat[:,i]-np.mean(dat[:,i])
            x2=dat[:,j]-np.mean(dat[:,j])
            xsuma=np.sum(x1*x2)
            mat[i,j]=xsuma/(len(columna)-1)

    return mat

matrizcov=pca2(datos,fila,columna)
matriznumpy=np.cov(np.transpose(datos))
#print(np.shape(matrizcov)) dimension de matriz 30,30
#print("-----------")
#print(matriznumpy)

#Calcule los autovalores y autovectores de la matriz de covarianza y los imprima (para esto puede usar los paquetes de linalg de numpy). Su mensaje debe indicar explıcitamente cual es cada autovector y su autovalor correspondiente.
resolviendo=np.linalg.eig(matrizcov)
eigvalues=resolviendo[0]
#print(len(eigvalues)) 30
eigvectors=resolviendo[1]
print("#1")
print("------------------------------------------------------------------")
for i in range(len(eigvalues)):
    print ("El valor propio", i+1," de la matriz de covarianza es",eigvalues[i])
    print ("Su vector asociado es",eigvectors[:,i])
#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
print("------------------------------------------------------------------")
print("#2")
print("Teniendo en cuenta los autovectores, los parametros mas importantes son los primeros dos, puesto que son aquellos que presentan los mayores autovalors y autovectores")
# teniendo en cuenta que np.eig nos da en orden de relevancia los eigen vectores y valores, para el ajuste, se utilizaran los primeros dos
eig1=eigvectors[:,0]
eig2=eigvectors[:,1]

#Haga una proyeccion de sus datos en el sistema de coordenadas PC1, PC2 y grafique estos datos. Use un color distinto para el diagnostico maligno y el benigno y la guarde dicha grafica sin mostrarla en ApellidoNombre_PCA.pdf.
PCA1=np.dot(datos,eig1)
PCA2=np.dot(datos,eig2)

def clasificacion(arr,p1,p2):
    bpc1=[]
    mpc1=[]
    bpc2=[]
    mpc2=[]
    for i in range(len(arr)):
        if(arr[i]=="M"):
            mpc1.append(p1[i])
            mpc2.append(p2[i])
        elif(arr[i]=="B"):
            bpc1.append(p1[i])
            bpc2.append(p2[i])

    return mpc1,mpc2,bpc1,bpc2

clasificando=clasificacion(info,PCA1,PCA2)
malp1=clasificando[0]
malp2=clasificando[1]
bnp1=clasificando[2]
bnp2=clasificando[3]

plt.figure()
plt.scatter(PCA1,PCA2, label="PCA",color="yellow")
plt.scatter(malp1,malp2,label="Maligno",color="red")
plt.scatter(bnp1,bnp2,label="Benigno",color="blue")
plt.title('Proyeccion de Datos en sistema de coordenadas PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc="best")
plt.show()
#plt.savefig("AvilaDario_PCA.pdf")

#Imprima un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion, si no sirve o si puede ayudar al diagnostico para ciertos pacientes, argumentando claramente su posicion
print("------------------------------------------------------------------")
print("El metodo de PCA resulta ...para hacer esta clasificacion puesto que al ver la grafica de los malignos y benignos, es evidente que")

import numpy as np
import matplotlib.pylab as plt
import math

#Almacene los datos del archivo WDBC.dat.
datos=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
columna=datos[:,0]
print (len(columna))
fila=datos[0,:]
print(len(fila))
info=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(0,1))
# Calcule, con su implementacion propia, la matriz de covarianza de los datos y la imprima
'''
def pca(dat,filas,columnas):
    mat=np.zeros((len(filas),len(filas)))
    for i in range(len(fila)):
        for j in range(len(fila)):
            for k in range(len(columna)):
                x1=dat[k,i]-np.mean(dat[:,i])
                x2=dat[k,j]-np.mean(dat[:,j])
                xsuma=x1*x2
                k+=xsuma
            mat[i,j]=xsuma/(len(columna)-1)
            #print (mat[i,j])
    return mat
'''
def pca2(dat,filas,columnas):

    mat=np.zeros((len(filas),len(filas)))

    for i in range(len(fila)):
        for j in range(len(fila)):

            x1=dat[:,i]-np.mean(dat[:,i])
            x2=dat[:,j]-np.mean(dat[:,j])
            xsuma=np.sum(x1*x2)
            mat[i,j]=xsuma/(len(columna)-1)

    return mat

'''pc3(dat.filas,columnas)
N=np.shape(dat)[1]
mat=np.zeros((len(filas),len(filas)))
for i in range(N)
    for j in range(N)
        mat[i,j]=np.sum((dat[:,i]-np.mean(dat[:,i])*(dat[:,j]-np.mean(dat[:,j]))))/len(columna)-1'''



matrizcov=pca2(datos,fila,columna)
matriznumpy=np.cov(np.transpose(datos))
print(matrizcov)
#print("-----------")
#print(matriznumpy)

#Calcule los autovalores y autovectores de la matriz de covarianza y los imprima (para esto puede usar los paquetes de linalg de numpy). Su mensaje debe indicar explıcitamente cual es cada autovector y su autovalor correspondiente.
resolviendo=np.linalg.eig(matrizcov)
eigvalues=resolviendo[0]
print("Los valores propios de la matriz de covarianza son")#,eigvalues)
eigvectors=resolviendo[1]
print("Los vectores propios correspondientes de la matriz de covarianza son")#,eigvectors)

#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
print("Teniendo en cuenta los autovectores, los parametros mas importantes son")
# teniendo en cuenta que np.eig nos da en orden de relevancia los eigen vectores y valores, para el ajuste, se utilizaran los primeros dos
eig1=eigvectors[:,0]
eig2=eigvectors[:,1]
#Haga una proyeccion de sus datos en el sistema de coordenadas PC1, PC2 y grafique estos datos. Use un color distinto para el diagnostico maligno y el benigno y la guarde dicha grafica sin mostrarla en ApellidoNombre_PCA.pdf.


'''
def clasificacion():
    bpc1=[]
    mpc2=[]
    bpc1=[]
    mpc2=[]
    return'''



plt.figure()
plt.plot(label="Maligno",color="red")
plt.plot(label="Benigno",color="blue")
plt.title('Proyeccion de Datos en sistema de coordenadas PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc="best")
plt.show()
#plt.savefig("")

#Imprima un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion, si no sirve o si puede ayudar al diagnostico para ciertos pacientes, argumentando claramente su posicion
print("El metodo de PCA resulta ...para hacer esta clasificacion")

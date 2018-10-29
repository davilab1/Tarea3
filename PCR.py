import numpy as np
import matplotlib.pylab as plt
import math

#Almacene los datos del archivo WDBC.dat.
datos=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30))
columna=datos[:,0]
print (len(columna))
fila=datos[0,:]
print(len(fila))
info=np.gefromtxt("WDBC.dat",delimiter=",",usecols=(1,2))
# Calcule, con su implementacion propia, la matriz de covarianza de los datos y la imprima

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


N=np.shape(dat)[1]
mat=np.zeros((len(filas),len(filas)))
for i in range(N)
    for j in range(N)
        mat[i,j]=np.sum((dat[:,i]-np.mean(dat[:,i])*(dat[:,j]-np.mean(dat[:,j]))))/len(columna)-1




matrizcov=pca(datos,fila,columna)
#print(matrizcov)
print(shape(matrizcov))
#Calcule los autovalores y autovectores de la matriz de covarianza y los imprima (para esto puede usar los paquetes de linalg de numpy). Su mensaje debe indicar explıcitamente cual es cada autovector y su autovalor correspondiente.
resolviendo=np.linalg.eig(matrizcov)
eigvalues=resolviendo[0]
print("Los valores propios de la matriz de covarianza son",eigvalues)
eigvectors=resolviendo[1]
print("Los vectores propios correspondientes de la matriz de covarianza son",eigvectors)

#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
print("Teniendo en cuenta los autovectores, los parametros mas importantes son")

#Haga una proyeccion de sus datos en el sistema de coordenadas PC1, PC2 y grafique estos datos. Use un color distinto para el diagnostico maligno y el benigno y la guarde dicha grafica sin mostrarla en ApellidoNombre_PCA.pdf.
nuevascoord= np.dot(datos.eigvectors)
'''

plt.figure()
plt.plot(label="Maligno",color="red")
plt.plot(label="Benigno",color="blue")
plt.title('Proyeccion de Datos en sistema de coordenadas PCA')
plt.xlabel()
plt.ylabel()
plt.savefig("")



#Imprima un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion, si no sirve o si puede ayudar al diagnostico para ciertos pacientes, argumentando claramente su posicion
print("El metodo de PCa resulta ...para hacer esta clasificacion")'''

#Paquetes que se importan para el correcto desarrollo del ejercicio
#Todo se corrio en python3
import numpy as np
import matplotlib.pylab as plt
import math

#Almacene los datos del archivo WDBC.dat.
# se guarda la matriz de 30  variables en un arreglo
datos=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
#para determinar el numero de columnas y de filas
columna=datos[:,0]
#print (len(columna)) 569
fila=datos[0,:]
#print(len(fila)) 30
#https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
#Como la segunda columna es un string se separa y se guarda en un arreglo diferente que se usara al final
info=np.genfromtxt("WDBC.dat",delimiter=",",usecols=(1),dtype="unicode")
#print(len(info))
#luego de analizar de manera critica los datos y viendo que estos no estaban dando correctamente a la hora de graficar, llegue a la conclusion y cai en cuenta de que no estabamos utilizando el indice de correlacion de pearson, el cual lo que hace es normalizar todos mis datos, puesto que no estamos teniedo en cuenta que todos estan en diferentes unidades. Asi, lo que se debe hacer es restar cada dato con la media de la columna y acto seguido, se debe dividir o en la raiz de la varianza o simplemente dividir en la desviacion estandar
#normalizando los datos
for i in range(len(fila)):
    datos[:,i]=(datos[:,i]-np.mean(datos[:,i]))/(np.std(datos[:,i]))

# Calcule, con su implementacion propia, la matriz de covarianza de los datos y la imprima
#implementacion de pca, en donde se utiliza la formula de la covarianza y al tiempo se va añadiendo en una matriz que fue creada ahi mismo
def pca2(dat,filas,columnas):

    mat=np.zeros((len(filas),len(filas)))

    for i in range(len(fila)):
        for j in range(len(fila)):

            x1=dat[:,i]-np.mean(dat[:,i])
            x2=dat[:,j]-np.mean(dat[:,j])
            xsuma=np.sum(x1*x2)
            mat[i,j]=xsuma/(len(columna)-1)

    return mat
# se evalua la funcion con los parametros dados
matrizcov=pca2(datos,fila,columna)
print("La matriz de covianza es la siguiente:")
print(matrizcov)
#se verifica que el valor que da el paquete de numpy sea el mismo que el de la implementacion propia, y efectivamente lo es
matriznumpy=np.cov(np.transpose(datos))
#print(np.shape(matrizcov)) dimension de matriz 30,30
#print(matriznumpy)
print("------------------------------------------------------------------")
#Calcule los autovalores y autovectores de la matriz de covarianza y los imprima (para esto puede usar los paquetes de linalg de numpy). Su mensaje debe indicar explıcitamente cual es cada autovector y su autovalor correspondiente.
#Calculando los autovalores y autovectores, por la literatura, se sabe que .eig da primero el autovalor y luego el autovector asociado
resolviendo=np.linalg.eig(matrizcov)
eigvalues=resolviendo[0]
#print(len(eigvalues)) 30
eigvectors=resolviendo[1]
print("#1")
print("------------------------------------------------------------------")
# como se pide que se muestre el autovalor y autovector asociado al mismo, se realiza un for que recorra esta cantidad y los imprima en orden
for i in range(len(eigvalues)):
    print ("El valor propio", i+1," de la matriz de covarianza es",eigvalues[i])
    print ("Su vector asociado es",eigvectors[:,i])
#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
print("------------------------------------------------------------------")
print("#2")
print("Teniendo en cuenta los autovalores y autovectores, los parametros mas importantes son los primeros dos, puesto que son aquellos que presentan los mayores autovalores y autovectores, de este modo, los parametros principales a evaluar en el Pca son PCA1 y PCA2, ya que son aquellos que presentan una mayor varicion con respecto a las demas variables.Es por esto que , se eligieron como los componenetes pricipales del PCA")
# teniendo en cuenta que np.eig nos da en orden de relevancia los eigen vectores y valores, para el ajuste, se utilizaran los primeros dos
eig1=eigvectors[:,0]
#print(eig1)
#print("----------------------")
eig2=eigvectors[:,1]
#print(eig2)
#print("----------------------")
#Haga una proyeccion de sus datos en el sistema de coordenadas PC1, PC2 y grafique estos datos. Use un color distinto para el diagnostico maligno y el benigno y la guarde dicha grafica sin mostrarla en ApellidoNombre_PCA.pdf.
# se utiliza np.dot para realizar la proyeccion
PCA1=np.dot(datos,eig1)
PCA2=np.dot(datos,eig2)
# se deben clasificar los datos en Mo B y se crea un metodo que recorre todas las filas y las clasifica
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
#Se separan en diferentes variables lo que me da el metodo, para poder graficarlo de manera mas facil
clasificando=clasificacion(info,PCA1,PCA2)
malp1=clasificando[0]
malp2=clasificando[1]
bnp1=clasificando[2]
bnp2=clasificando[3]

plt.figure()
#plt.scatter(PCA1,PCA2, label="PCA",color="yellow")
plt.scatter(malp1,malp2,label="Maligno",color="red")
plt.scatter(bnp1,bnp2,label="Benigno",color="blue")
plt.title('Proyeccion de Datos en sistema de coordenadas PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc="best")
plt.savefig("AvilaDario_PCA.pdf")
plt.show()

#Imprima un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion, si no sirve o si puede ayudar al diagnostico para ciertos pacientes, argumentando claramente su posicion
print("------------------------------------------------------------------")
print("#3")
print("El metodo de PCA resulta bastante eficiente a la hora de ver el area y las variables las cuales representan a los casos malignos y benignos. Es posible ver una clara diferencia de estos. A la hora de ver la grafica es posible observar en un cierto rango de valores bajos tanto de PC1 y PC2 se agrupan los casos benignos y los casos malignos pero los casos malignos estan mucho mas  dispersos a los largo de la grafica, de este modo se podria pensar que se clasificaron de manera correcta. Para los casos Benignos, estos se encuentran mejor corelacionados con PC2 que con PC1, mientras que por otra parte los casos malignos estan mejor correlacionados con PC1 que con PC2, es por esto que se puede ver una diferencia entre los datos agrupados cuando se grafican de manera simultanea.Por lo tanto,es posible afirmar que el pca que se realizo dio una correcta correlacion entre las variables para dar un buen diagnostico y automatizar el proceso. No obstante, cabe resaltar que se tuvo en cuenta una de las debilidades que presenta la matriz de covarianza como metodo de correlacion y es que no se tienen en cuenta las unidades de las variables que se estan procesando, es por esto que se normalizaron los datos antes de comenzar, lo cual ayudo a que los datos se distribuyeran mejor y se viera graficamente las dos zonas en las cuales se presentan los datos.")

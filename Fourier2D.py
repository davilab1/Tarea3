#Paquetes que se importan para el correcto desarrollo del ejercicio
#Todo se corrio en python3
import numpy as np
import matplotlib.pylab as plt
import math
import scipy as sp
from scipy import ndimage
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift
from matplotlib.colors import LogNorm

#Almacene la arboln arbol.png en una arreglo de numpy.
arbol=plt.imread('arbol.png')
# Tomando como referencia https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
'''
plt.figure()
plt.imshow(arbol,plt.cm.gray)
plt.title('prueba')
'''
#Usando los paquetes de scipy, realice la transformada de Fourier de la arboln. Eligiendo una escala apropiada
transfIm=fft2(arbol)
#print(transfIm)
frecu=fftfreq(256)
#centrando la transformada
transhiffIm=fftshift(transfIm)

'''
plt.figure()
plt.plot(frecu,np.abs(transfIm))
plt.title('probando para ver frecuencias')
plt.show()'''
#print(np.shape(transhiffIm)) 256x256
# haga una grafica de dicha transformada y guardela sin mostrarla en ApellidoNombre_FT2D.pdf.
plt.figure()
plt.imshow(np.log(abs(transhiffIm)),plt.cm.gray)#realizando esta grafica, y viendo la escala de colores, se determina ruido como valor de 7.5
#plt.imshow(abs(transhiffIm),plt.cm.gray)
plt.title('Transformada de fourier para arbol')
plt.colorbar()
plt.savefig("AvilaDario_FT2D.pdf")

#Haga un filtro que le permita eliminar el ruido periodico de la arbol. Para esto haga pruebas de como debe modificarla transformada de Fourier.

# este error me ayudo a modificar el filtro ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
#creando el metodo que filtra, al graficar "probando para ver frecuencias, fue posible observar unos picos en un rando entre 4110 y 5000 , es por esto que en esos valores se colocaron 0"
def filtrando(ftrans):
    for i in range(len(ftrans)):
        for j in range(len(ftrans)):
            if (abs(ftrans[i,j])>4110.0 and abs(ftrans[i,j])<5000.0 ):#son los valores en donde se encuentran los picos(ruido) formada
                ftrans[i,j]=0
            else:
                ftrans[i,j]=ftrans[i,j]
    return ftrans

filtroTransf=filtrando(transhiffIm)

#Grafique la transformada de Fourier despues del proceso de filtrado, esta vez en escala LogNorm y guarde dicha grafica
#sin mostrarla en ApellidoNombre_FT2D_filtrada.pdf.

plt.figure()
plt.imshow(np.abs(filtroTransf),norm=LogNorm(vmin=5),cmap='gray')
plt.title('Transformada de Fourier despues de Filtro')
plt.colorbar()
plt.savefig("AvilaDario_FT2D_Filtrada.pdf")


#Haga la transformada de Fourier inversa y grafique la arboln filtrada. Verifique que su filtro elimina el ruido periodico
# y guarde dicha arboln sin mostrarla en ApellidoNombre_arboln_filtrada.pdf.
# se hace la inversa del fiiltro centrado
newarbol=ifft2(fftshift(filtroTransf))
arbbbbol=ifft2(filtroTransf)

#File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/matplotlib/image.py", line 642, in set_dataraise TypeError("Image data cannot be converted to float")
#Se pasa a imagen la inversa
plt.figure()
plt.title('Transformada inversa de Fourier filtrada')
plt.imshow(np.abs(arbbbbol),plt.cm.gray)
plt.savefig("AvilaDario_arboln_filtrada.pdf")
plt.show()
#plt.imshow(imfiltrada, cmap='gray')

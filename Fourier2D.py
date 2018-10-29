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
#plt.figure()
#plt.imshow(arbol,plt.cm.gray)
#plt.title('prueba')

#Usando los paquetes de scipy, realice la transformada de Fourier de la arboln. Eligiendo una escala apropiada
transfIm=fft2(arbol)

transhiffIm=fftshift(transfIm)

# haga una grafica de dicha transformada y guardela sin mostrarla en ApellidoNombre_FT2D.pdf.
plt.figure()
plt.imshow(np.log(abs(transfIm)),plt.cm.gray)#realizando esta grafica, y viendo la escala de colores, se determina ruido como valor de 7.5
#plt.imshow(abs(transhiffIm),plt.cm.gray)
plt.title('Transformada de fourier para arbol')
plt.colorbar()
#hola

#plt.savefig("AvilaDario_FT2D.pdf)

#Haga un filtro que le permita eliminar el ruido periodico de la arbol. Para esto haga pruebas de como debe modificar
#la transformada de Fourier.

def filtrando(ftrans):
    for i in range(len(ftrans)):
        for j in range(len(ftrans)):
            if (np.log(ftrans[i,j])>7.5):
                ftrans[i,j]==0
            else:
                ftrans[i,j]=ftrans[i,j]
    return ftrans

filtroTransf=filtrando(transhiffIm)

#Grafique la transformada de Fourier despues del proceso de filtrado, esta vez en escala LogNorm y guarde dicha grafica
#sin mostrarla en ApellidoNombre_FT2D_filtrada.pdf.
plt.figure()
plt.imshow(np.abs(filtroTransf),norm=LogNorm(vmin=5))
plt.title('Transformada de Fourier despues de Filtro')

#plt.savefig("AvilaDario_FT2D_Filtrada.pdf)

#Haga la transformada de Fourier inversa y grafique la arboln filtrada. Verifique que su filtro elimina el ruido periodico
# y guarde dicha arboln sin mostrarla en ApellidoNombre_arboln_filtrada.pdf.
newarbol=ifft2(fftshift(filtroTransf))

#tinvf2=ifftshift(filtroTransf)
#imfiltrada=ifft2(tinvf2).real
#File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/matplotlib/image.py", line 642, in set_dataraise TypeError("Image data cannot be converted to float")
plt.figure()
plt.title('Transformada inversa de Fourier filtrada')
plt.imshow(np.abs(newarbol),plt.cm.gray)
#plt.imshow(imfiltrada, cmap='gray')
plt.show()
#plt.savefig("AvilaDario_arboln_filtrada.pdf)

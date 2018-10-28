import numpy as np
import matplotlib.pylab as plt
import math
import scipy as sp
from scipy import ndimage
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2, fftshift, ifftshift
from matplotlib.colors import LogNorm

#Almacene la arboln arbol.png en una arreglo de numpy.
arbol=plt.imread('arbol.png')

'''
plt.figure()
plt.plot(arbol)
plt.title('prueba')
plt.show()
'''
#Usando los paquetes de scipy, realice la transformada de Fourier de la arboln. Eligiendo una escala apropiada
transfIm=fft2(arbol)
transhiffIm=fftshift(transfIm)

# haga una grafica de dicha transformada y guardela sin mostrarla en ApellidoNombre_FT2D.pdf.
plt.figure()
# Tomando como referencia https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
plt.imshow(abs(transhiffIm),vmin=5)
plt.imshow(abs(transhiffIm),plt.cm.gray)
#plt.imshow(arbol)
plt.title('Transformada de fourier para arbol')
plt.colorbar()
plt.show()
#plt.savefig("AvilaDario_FT2D.pdf)

#Haga un filtro que le permita eliminar el ruido periodico de la arbol. Para esto haga pruebas de como debe modificar
#la transformada de Fourier.

	for i in range(len(transfIm):
		if(abs(freq[i])>2000.0):
			ftrans[i]=0
		else:
			ftrans[i]=ftrans[i]
 	    return ftrans


#Grafique la transformada de Fourier despues del proceso de filtrado, esta vez en escala LogNorm y guarde dicha grafica
#sin mostrarla en ApellidoNombre_FT2D_filtrada.pdf.
plt.figure()
plt.plot()
plt.yscale('log')
plt.title('Transformada de Fourier despues de Filtro')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
#plt.savefig("AvilaDario_FT2D_Filtrada.pdf)


#Haga la transformada de Fourier inversa y grafique la arboln filtrada. Verifique que su filtro elimina el ruido periodico
# y guarde dicha arboln sin mostrarla en ApellidoNombre_arboln_filtrada.pdf.

tinvf2=ifftshift(filtrada)
imfiltrada=ifft2(tinvf2).real

plt.figure()
plt.title('Transformada inversa de Fourier filtrada')
plt.imshow(imfiltrada, cmap='gray')
#plt.savefig("AvilaDario_arboln_filtrada.pdf)
'''

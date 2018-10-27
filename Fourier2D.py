import numpy as np
import matplotlib.pylab as plt
import math
import scipy as sp
from scipy.fftpack import fft, fftfreq


#Almacene la imagen arbol.png en una arreglo de numpy.


#Usando los paquetes de scipy, realice la transformada de Fourier de la imagen. Eligiendo una escala apropiada
# haga una grafica de dicha transformada y guardela sin mostrarla en ApellidoNombre_FT2D.pdf.
plt.figure()
plt.plot()
plt.title('')
plt.show()
plt.xlabel()
plt.ylabel()
plt.savefig("AvilaDario_FT2D.pdf)

#Haga un filtro que le permita eliminar el ruido periodico de la imagen. Para esto haga pruebas de como debe modificar
#la transformada de Fourier.


#Grafique la transformada de Fourier despues del proceso de filtrado, esta vez en escala LogNorm y guarde dicha grafica
#sin mostrarla en ApellidoNombre_FT2D_filtrada.pdf.
plt.figure()
plt.plot()
plt.title('')
plt.show()
plt.xlabel()
plt.ylabel()
plt.savefig("AvilaDario_FT2D_Filtrada.pdf)


#Haga la transformada de Fourier inversa y grafique la imagen filtrada. Verifique que su filtro elimina el ruido periodico
# y guarde dicha imagen sin mostrarla en ApellidoNombre_Imagen_filtrada.pdf.

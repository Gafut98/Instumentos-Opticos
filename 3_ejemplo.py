import matplotlib.pyplot as plt
import numpy as np
import cv2

#carga la imágen
imagen_fuente_1 = plt.imread('airplane.jpg',cv2.IMREAD_GRAYSCALE)

#convertir la imagen a arreglo tipo float
imagen_fuente_1_float = np.asarray(imagen_fuente_1,dtype=np.float32)

#norm="ortho": modo normalizado. Escala TF * 1/(sqrt(Weigth x Height)
#transformada de fourier de la imagen 1
FFT_imagen_1 = np.fft.fft2(imagen_fuente_1_float,norm="ortho")
FFT_imagen_centrada_1=np.fft.fftshift(FFT_imagen_1)

#espectro magnitud con compresión del rango dinámico
#para la mejor visualización
magnitudFFT_1 = np.log(1+np.abs(FFT_imagen_centrada_1))

#crea una mascara que servirá como filtro de frecuencias en el espacio de Fourier
mask2=np.ones(np.shape(imagen_fuente_1),dtype="uint8")

coordx=int(np.shape(imagen_fuente_1)[1]/2)
coordy=int(np.shape(imagen_fuente_1)[0]/2) 

#tamaño del disco, ri=7, re=15
cv2.circle(mask2,((coordx),(coordy)),15,0,-1) #Circulo externo
cv2.circle(mask2,((coordx),(coordy)),7,1,-1) #Circulo interno


#producto de la transformada con la mascara creada
producttt=FFT_imagen_centrada_1*mask2

#magnitud del producto
magnitudFFT_p = np.log(1+np.abs(producttt))

#transformada inversa del producto
IFFT_producto = np.fft.ifft2(producttt,norm="ortho")
imagen_final = np.real(IFFT_producto)
                                  

#preparar para desplegar resultados
f = plt.figure(figsize=(10,10))
ax1 = f.add_subplot(2,2, 1)
ax2 = f.add_subplot(2,2, 2)
ax3 = f.add_subplot(2,2, 3)
ax4 = f.add_subplot(2,2, 4)

f.suptitle("Ejemplo de aplicación del montaje propuesto",fontsize=12)

plt.rcParams.update({'font.size':8})

ax1.set_title("a. Imagen de entrada")
ax1.imshow(imagen_fuente_1, cmap='gray')
#apagar ejes
ax1.axis('off')

ax2.set_title("b. Magnitud de la imagen de entrada en el espacio de frecuencias")
ax2.imshow(magnitudFFT_1, cmap='gray')
#apagar ejes
ax2.axis('off')


ax4.set_title("d. Imagen con frecuencias espaciales filtradas")
ax4.imshow(np.abs(imagen_final), cmap='gray')
#apagar ejes
ax4.axis('off')

ax3.set_title("c. Magnitud del producto")
ax3.imshow(magnitudFFT_p, cmap='gray')
#apagar ejes
ax3.axis('off')

f.canvas.set_window_title('Montaje propuesto')

plt.show()

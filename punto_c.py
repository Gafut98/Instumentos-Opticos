import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

##Primero se procesan las dos imagenes (la completa y la pista), pasandolas a blanco y negro

#Valor que define si se convierte a blanco o negro cada pixel
b=110

c = cv2.imread("Equipo 01/c.jpg", cv2.IMREAD_GRAYSCALE) #lectura en escala de grises de la imagen
c_clue=cv2.imread("Equipo 01/c_clue.jpg", cv2.IMREAD_GRAYSCALE) #lectura en escala de grises de la pista

#conversión a blanco y negro
(thresh, blackAndWhiteImage) = cv2.threshold(c, b, 255, cv2.THRESH_BINARY) 
(thresh, blackAndWhiteImage2) = cv2.threshold(c_clue, b, 255, cv2.THRESH_BINARY)


img = blackAndWhiteImage
img2 = img.copy()
template = blackAndWhiteImage2
w, h = template.shape[::-1]

# Se elige el método de matching, que corresponde a la correlación
method = eval('cv.TM_CCOEFF')

#Se aplica el método
res = cv.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

#Se pondrá un rectangulo negro al rededor de la imagenque señale el resultado
cv.rectangle(img,top_left, bottom_right, (0,0,0), 2)

#desplegar resultados
fig, axes = plt.subplots(2, 1, figsize=(20, 50))
ax = axes.ravel()

#tamaño de letra de los títulos
plt.rcParams.update({'font.size': 6}) 

ax[0].set_title('Matching Result')
ax[0].imshow(res,cmap = 'gray')
ax[0].set_yticklabels([])#no mostrar ticks
ax[0].set_xticklabels([])#no mostrar ticks


ax[1].set_title('Detected Point')
ax[1].imshow(img, cmap='gray')
ax[1].set_yticklabels([])#no mostrar ticks
ax[1].set_xticklabels([])#no mostrar ticks

plt.show()

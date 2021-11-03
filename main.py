import numpy as np
import  cv2
import matplotlib.pyplot as plt
imagen=cv2.imread("b.png",0)
cv2.imshow("Imagen Abierta",imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plt.imshow(imagen,cmap="Greys")

#plt.show()

#pip install cv2
#new_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])

#print(new_matrix)




#nm=1e-9
#um=1e-6
#mm=1e-3

#wave_length=650*nm
#wave_range=np.arange(0,5*um,10*nm) #espacio discreto de x,0 hasta 5 micras paso de 10 nanometros
#k=2*np.pi/wave_length

#onda1D=np.sin(k*wave_range)

#plt.plot(onda1D)
#plt.show()
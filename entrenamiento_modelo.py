from data import celsius, fahrenheit
from myFirstNeuronalRed import modelo
import matplotlib.pyplot as plt
#Entrenamiento
print('Se inicia el entrenamiento...')
historial = modelo.fit(celsius,fahrenheit, epochs = 750, verbose = False)
print('Modelo entrenado')

def grafica_resultado_entrenamiento(historial):
    plt.xlabel('Epochs')
    plt.ylabel('Magnitud de perdida')
    plt.plot(historial.history['loss'])
    return plt.show()

grafica_resultado_entrenamiento(historial) #Grafica de perdida
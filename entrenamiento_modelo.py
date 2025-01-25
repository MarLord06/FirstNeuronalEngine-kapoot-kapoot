
import matplotlib.pyplot as plt
#Entrenamiento
def entrenamiento_modelo(modelo, celsius, fahrenheit):
    print('Se inicia el entrenamiento...')
    historial = modelo.fit(celsius,fahrenheit, epochs = 1000, verbose = False)
    print('Modelo entrenado')
    return historial

def grafica_resultado_entrenamiento(historial):
    plt.xlabel('Epochs')
    plt.ylabel('Magnitud de perdida')
    plt.plot(historial.history['loss'])
    return plt.show()

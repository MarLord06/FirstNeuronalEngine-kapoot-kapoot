from data import celsius, fahrenheit
from myFirstNeuronalRed import modelo

#Entrenamiento
print('Se inicia el entrenamiento...')
historial = modelo.fit(celsius,fahrenheit, epochs = 750, verbose = False)
print('Modelo entrenado')

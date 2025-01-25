import os
import tensorflow as tf
from myFirstNeuronalRed import pruebas, capa_neuronal, compilacion_modelo
from entrenamiento_modelo import entrenamiento_modelo, grafica_resultado_entrenamiento
from data import celsius, fahrenheit
modelo_ruta = 'modelo_entrenado.keras'

if os.path.exists(modelo_ruta):
    modelo = tf.keras.models.load_model(modelo_ruta)
    print('Modelo cargado')
else:
    modelo = capa_neuronal(1,1)  # 1 neurona, 1 forma
    modelo = compilacion_modelo(modelo)
    historial = entrenamiento_modelo(modelo, celsius, fahrenheit)
    grafica_resultado_entrenamiento(historial)
    modelo.save(modelo_ruta)
    print('modelo entrenado y guardado en el archivo modelo_entrenado.keras')

celsius = float(input('Ingrese los grados Celsius a convertir: '))
print(f'La prediccion de mi red es: {round(pruebas(modelo, celsius))} fahrenheit') #Debe dar 100*1.8+32 = 212.0

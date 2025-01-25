import tensorflow as tf
import numpy as np

print("TensorFlow version: " + tf.__version__)


# Funcion para creacion de capa neuronal
def capa_neuronal(densidad, forma):
    capa = tf.keras.layers.Dense(units=densidad, input_shape=[forma])
    modelo = tf.keras.sequential([capa])
    return modelo

modelo = capa_neuronal(1,1)  # 1 neurona, 1 forma

#Compilacion y preparacion del modelo
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)




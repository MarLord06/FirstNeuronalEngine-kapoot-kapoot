import tensorflow as tf
import numpy as np

print("TensorFlow version: " + tf.__version__)


# Funcion para creacion de capa neuronal
def capa_neuronal(densidad, entradas):
    input_layer = tf.keras.layers.Input(shape=[entradas])
    dense_layer = tf.keras.layers.Dense(units=densidad)(input_layer)
    modelo = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
    return modelo


#Compilacion y preparacion del modelo
def compilacion_modelo(modelo):
    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(0.1),
        loss = 'mean_squared_error'
    )
    return modelo


def pruebas(modelo, celsius):
    return modelo.predict([np.array([celsius])])[0][0]


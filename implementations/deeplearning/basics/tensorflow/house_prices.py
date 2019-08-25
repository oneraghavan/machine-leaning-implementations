import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1, 2, 3, 4.0, 5, 6, 8, 9, 10], dtype=float)
ys = np.array([100, 150, 200, 250, 300, 350, 450, 500, 550], dtype=float)
model.fit(xs, ys)
print(model.predict([7.0]))

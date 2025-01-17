import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


customCallback = CustomCallback()


def train_mnist():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(len(x_train))
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[customCallback])
    # model fitting
    return history.epoch, history.history['acc'][-1]


train_mnist()

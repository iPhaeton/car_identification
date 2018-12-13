from keras.layers import Input, LSTM, BatchNormalization, Dense
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

class LSTMDecoder:
    def __init__(self, X_train, y_train, X_dev, y_dev, lr=0.00001, weights_path = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.y_train_hot = to_categorical(self.y_train) if y_train is not None else None
        self.y_dev_hot = to_categorical(self.y_dev)

        input_shape = self.X_dev.shape[1:]
        output_shape = self.y_dev_hot.shape[1]

        inputs = Input(shape=input_shape)
        X = BatchNormalization()(inputs)
        X = LSTM(1024)(X)
        outputs = Dense(output_shape, activation='softmax')(X)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.train.AdamOptimizer(lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
        )

        if weights_path != None:
            model.load_weights(weights_path)

        self.model = model

    def train(self, epochs, callbacks, batch_size=128):
        self.model.fit(
            self.X_train,
            self.y_train_hot,
            batch_size=batch_size,
            epochs=epochs,
            verbose=True,
            callbacks=callbacks,
            validation_data=(self.X_dev, self.y_dev_hot)
        )

    def predict(self, mode):
        X = self.X_dev if mode == 'dev' else self.X_train
        predictions = self.model.predict(X, verbose = True)
        return np.argmax(predictions, axis=1), predictions
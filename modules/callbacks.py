import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model_interface = None, true_classes = None):
        self.model_interface = model_interface
        self.true_classes = true_classes

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.target_val_acc = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
        self.ax1 = ax1
        self.ax2 = ax2
    
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.i += 1

        if (self.model_interface is not None) & (self.true_classes is not None):
            print('Validating...')
            predictions, _ = self.model_interface.predict('dev')
            target_acc = np.sum(predictions == self.true_classes) / predictions.shape[0]
            self.target_val_acc.append(target_acc)
        
        clear_output(wait=True)
        self.ax1.plot(self.x, self.losses, label="loss")
        self.ax1.plot(self.x, self.val_losses, label="val loss")
        self.ax1.legend()
        self.ax2.plot(self.x, self.acc, label="accuracy")
        self.ax2.plot(self.x, self.val_acc, label="val accuracy")
        if (len(self.target_val_acc) != 0):
            self.ax2.plot(self.x, self.target_val_acc, label="target val acc")
        self.ax2.legend()
        plt.show()

        for i in range(self.i):
            print('Epoch ' + str(i+1))
            print('-----------------------')
            print('- Loss:', self.losses[i])
            print('- Accuracy:', self.acc[i])
            print('- Validation loss:', self.val_losses[i])
            print('- Validation accuracy:', self.val_acc[i])
            print('- Target validation accuracy:', self.target_val_acc[i])
            print(' ')

class OptimizationHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))

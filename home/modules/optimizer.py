import numpy as np

class Optimizer:
    def __init__(self, Constructor, **kwargs):
        self.Constructor = Constructor
        self.kwargs = kwargs
        self.history = {}
        self.current = None

    def update_history(self, i, j, k, lr, reg, callbacks):
        self.history[f'model-{i},initializer-{j},regularizer-{k},lr-{lr},reg-{reg}'] = np.stack([
            callbacks.val_losses,
            callbacks.val_acc,
        ], axis=1)

    def optimize(self, models, initializers, regularizers, learning_rates, reg_strengths, epochs, ranges_size, callbacks=None):
        learning_rates = 10 ** np.random.uniform(learning_rates[0], learning_rates[1], ranges_size)
        learning_rates = np.sort(learning_rates)
        reg_strengths = 10 ** np.random.uniform(reg_strengths[0], reg_strengths[1], ranges_size)
        reg_strengths = np.sort(reg_strengths)
        print(learning_rates, reg_strengths)
        for i, model in enumerate(models):
            for j, initializer in enumerate(initializers):
                for k, regularizer in enumerate(regularizers):
                    for lr in learning_rates:
                        for reg in reg_strengths:
                            self.current = self.Constructor(
                                hidden_layers=model, 
                                kernel_initializer=initializer, 
                                output_regularizer=regularizer,
                                lr=lr,
                                reg=reg,
                                **self.kwargs,
                            )
                            self.current.train(epochs, [callbacks])
                            self.update_history(i, j, k, round(lr, 4), round(reg, 4), callbacks)
        return self.history

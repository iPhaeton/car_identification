from keras.layers import AveragePooling2D, MaxPooling2D
from keras import Model

def get_base_model(model_constructor, preprocessor, img_size, pooling=None, verbose=True, layer=None):    
    base_model = model_constructor(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3), pooling=pooling)
    
    model = None
    if layer == None:
        model = base_model
    else:
        output = base_model.layers[layer].output
        
        if pooling != None:
            output_shape = base_model.layers[layer].output_shape
            Pooling = AveragePooling2D if pooling == 'avg' else MaxPooling2D
            output = Pooling(pool_size=(output_shape[1], output_shape[2]))(output)
            
        model = Model(input=base_model.input, output=output)
    
    if verbose == True:
        model.summary()
        
    return model
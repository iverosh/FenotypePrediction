import keras


def MyModel(num_circuit_layers, IMG_SIZE):
    input_shape=(IMG_SIZE, IMG_SIZE, 1)

    num_qubits = 4

    filters = [4] * num_circuit_layers # Same number of filters for each circuit layer

    kernel_sizes = [3] * num_circuit_layers # Same kernel size for each circuit layer

    strides = [1] * num_circuit_layers # Same stride for each circuit layer

    # quantum circuit layers

    circuit_layers = []

    for f, k, s in zip(filters, kernel_sizes, strides):

        circuit_layers.append(keras.layers.Convolution2D(filters = f, kernel_size = k, strides = s, padding = "same", activation= "tanh"))

        circuit_layers.append(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # classical layers

    dense_layers =[keras.layers.Flatten(), 
                   keras.layers.Dense(128, activation="relu"),
                   keras.layers.Dense(64, activation="relu"),
                   keras.layers.Dense(4, activation="softmax")]

    # Combine circuit and dense layers

    model=keras.models.Sequential([keras.layers.Input(shape=input_shape), 
                                  *circuit_layers, *dense_layers])
    
    opt = keras.optimizers.SGD(lr=0.01)

    model.compile(optimizer = opt,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"],
                  )

    return model
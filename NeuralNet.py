from tensorflow.keras import Model, layers


class NeuralNet(Model):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        # First Convolutional Block
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3),
                                   activation="relu", padding="same")
        self.pool1 = layers.MaxPool2D(strides=(2, 2))

        # Second Convolutional Block
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        self.pool2 = layers.MaxPool2D(strides=(2, 2))

        # Third Convolutional Block
        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
        self.pool3 = layers.MaxPool2D(strides=(2, 2))

        # fully-connected layer set
        self.flatten = layers.Flatten()
        self.dense5 = layers.Dense(units=512, activation="relu")
        self.do5 = layers.Dropout(0.5)

        # Classifier Head
        self.dense6 = layers.Dense(units=num_classes, activation="softmax")


    def call(self, inputs):
        # conv1
        x = self.conv1(inputs)
        x = self.pool1(x)

        # conv2
        x = self.conv2(x)
        x = self.pool2(x)

        # conv3
        x = self.conv3(x)
        x = self.pool3(x)

        # fully connected layer set
        x = self.flatten(x)
        x = self.dense5(x)
        x = self.do5(x)

        # classifier head
        x = self.dense6(x)

        return x

    def comp_and_fit(self, x_train, y_train, callbacks):
        self.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        history = self.fit(x=x_train,
                           y=y_train,
                           epochs=5,
                           batch_size=128,
                           validation_split=0.2,
                           callbacks=[callbacks])
        return history

    def eval(self, x_test, y_test):
        self.evaluate(
            # Fill with args
        )



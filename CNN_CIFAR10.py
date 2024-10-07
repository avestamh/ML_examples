import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

## Load and preprocess CIFAE-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0  # Normalize pixel values

# Class labels in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape = (32,32,3)), 
                     ##In CIFAR-10, images are 32x32 pixels with 3 color channels (Red, Green, and Blue)                
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  ## 10 class for CIFAR-10

])

# Compile the model
model.compile(optimizer='adam',
            #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Automatically applies softmax for logits and you  
                                                                                    # do not need explicitly include softmax in the last layer
             
            loss='sparse_categorical_crossentropy',  # No need for from_logits=True when using softmax
            metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')


### Make prediction on a single test image

def predict_single_image(index):
    ## pick a sample from the test dataset
    img = x_test[index]

    # display the image
    plt.imshow(img)
    plt.title(f'True label: {class_names[y_test[index][0]]}')
    plt.show()

    # Add a batch dimension (since the model expects batches of images)
    img = np.expand_dims(img, axis=0)

    ## make the prediction
    predictions = model.predict(img)  # The output is already probabilities due to softmax

    # find the class with highest probabilities
    predict_class = np.argmax(predictions)

    # print the results

    print(f'predicted Label: {class_names[predict_class]}')
    print(f'prediction probabilities: {predictions[0]}') ## Print the probabilities for all classes

# Predict and visualize the result for a single image
predict_single_image(0)  # You can change the index to test different images

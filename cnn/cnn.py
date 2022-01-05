import tensorflow as tf                         # importing library     
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# importing data set
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


# model building   and data cleansing*****************************************************************************************
model = models.Sequential()

# applying filter to input layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2, 2)))                                        
model.add(layers.Conv2D(64, (3, 3), activation='relu'))                       
model.add(layers.MaxPooling2D((2, 2)))                                        
model.add(layers.Conv2D(64, (3, 3), activation='relu'))                       
model.summary()

#flatten function 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#output layer
model.add(layers.Dense(10, activation='softmax')) 
model.summary()

#loading cifar10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape((50000, 32,32, 3))

#scaling data
train_images = train_images.astype('float32') / 255 
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# compiling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=7, batch_size=100)

#model evaluation for test
# test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
test_acc

# train accuracy
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
train_acc


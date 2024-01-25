#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os


# In[ ]:


# Define the path to the train folder containing images of dogs and cats
train_data_dir = 'cat_dog/sample/train'
# Define the path to the val folder containing images of dogs and cats
validation_data_dir = 'cat_dog/sample/validation'
# Define the path to the val folder containing images of dogs and cats
test_data_dir = 'cat_dog/sample/test'

# Set up data augmentation
train_datagen = image.ImageDataGenerator(rescale=1./255) 

# Load and augment the training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    class_mode='categorical'
)

# Set up data augmentation
validation_datagen = image.ImageDataGenerator(rescale=1./255)

# Load and augment the training dataset
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    class_mode='categorical'
)

# Set up data augmentation
test_datagen = image.ImageDataGenerator(rescale=1./255)

# Load and augment the training dataset
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    class_mode='categorical'
)


# In[ ]:


# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) #have to give the size of the original image as values 150, 150 ,if 3 => RGB ,if 1=> grayscale
model.add(MaxPooling2D(2, 2)) 
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten()) #convert any dimension of array to one dimension
model.add(Dense(128, activation='relu')) #this layer can only identify number
model.add(Dense(2, activation='softmax')) #binary classification
#can also use
#model.add(Dense(1, activation='sigmoid'))


# In[4]:


# compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# train the model
history = model.fit(train_generator, epochs=30, validation_data=validation_generator) #use the predifined 'validation_generator' as the validation_data


# In[ ]:


# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
#can change epochs, learning rate, Dense  layer =>128


# In[ ]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')


# In[ ]:


directory_path = 'cat_dog/sample/predict'
# List all files in the specified directory
files = os.listdir(directory_path)

# Iterate through the files and move those starting with 'dog_' to the 'dog' folder
for file in files:
    image_path = os.path.join(directory_path, file)
    
    if os.path.isfile(os.path.join(directory_path, file)):
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to match the model's expected input shape
        img_array /= 255.0  # Normalize pixel values to be in the range [0, 1]

        # Make a prediction
        prediction = model.predict(img_array)
        print(prediction)

        # `predictions` is a numpy array containing the predicted probabilities for each class
        # You can further process these predictions based on your specific use case
        # Assuming you have `predictions` from the previous code

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)

        # Define your class labels
        class_labels = ['cat', 'dog']

        # Get the corresponding class label
        predicted_class_label = class_labels[predicted_class_index]

        # Get the confidence score for the predicted class
        confidence_score = prediction[0, predicted_class_index]

        # Display the results
        print(f'Predicted Class: {predicted_class_label}')
        print(f'Confidence Score: {confidence_score * 100:.2f}%')


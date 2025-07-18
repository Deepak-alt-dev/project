import numpy as np import os import tensorflow as tf from tensorflow.keras import layers, models from sklearn.model_selection import train_test_split 
from 	tensorflow.keras.preprocessing.image 	import 
ImageDataGenerator 
 
# Constants 
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224  # Input image 
size 
BATCH_SIZE = 32 
EPOCHS = 50 
DATA_DIR = 'path_to_your_dataset'  # Update with your dataset path 
 
# Data Preparation def load_data(data_dir): classes = os.listdir(data_dir) images = [] 
labels = [] 
 
for class_label in classes: class_dir = os.path.join(data_dir, class_label) for img_file in os.listdir(class_dir): 
img_path = os.path.join(class_dir, img_file) img 	= 	tf.keras.preprocessing.image.load_img(img_path, 
target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)) 
img_array = tf.keras.preprocessing.image.img_to_array(img) images.append(img_array) 
labels.append(classes.index(class_label)) 
 
images = np.array(images) 
labels = np.array(labels) 
 
# Normalize images 
images = images / 255.0 
 
return images, labels 
 
# Load dataset 
images, labels = load_data(DATA_DIR) 
X_train, X_test, y_train, y_test = train_test_split(images, labels, 
test_size=0.2, random_state=42) 
 
# Data augmentation 
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest') 
 
# Fractal Convolutional Layer 
class FractalConvLayer(layers.Layer): def _init_(self, filters, kernel_size, **kwargs): super(FractalConvLayer, self)._init_(**kwargs) 
self.conv1 	= 	layers.Conv2D(filters, 	kernel_size, 
padding='same', activation='relu') 
self.conv2 	= 	layers.Conv2D(filters, 	kernel_size, 
padding='same', activation='relu') 
self.conv3 	= 	layers.Conv2D(filters, 	kernel_size, 
padding='same', activation='relu') 
 
def call(self, inputs): x1 = self.conv1(inputs) x2 = self.conv2(inputs) x3 = self.conv3(x1) 
return layers.add([x2, x3]) 
 
# Model Definition def create_fractal_cnn(): model = models.Sequential() 
 
model.add(FractalConvLayer(32, 	(3, 	3), 
input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))) 
model.add(layers.MaxPooling2D((2, 2))) 
 
model.add(FractalConvLayer(64, (3, 3))) model.add(layers.MaxPooling2D((2, 2))) 
 
model.add(FractalConvLayer(128, (3, 3))) 
model.add(layers.MaxPooling2D((2, 2))) 
 
model.add(layers.Flatten()) model.add(layers.Dense(512, activation='relu')) model.add(layers.Dropout(0.5)) model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))  # Number of classes 
 
return model 
 
# Compile and Train the Model model = create_fractal_cnn() model.compile(optimizer='adam', 
loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
 
# Fit the model 
history 	= 	model.fit(train_datagen.flow(X_train, 	y_train, 
batch_size=BATCH_SIZE), 
epochs=EPOCHS, 
validation_data=(X_test, y_test)) 

# Evaluate the model 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2) print(f'\nTest accuracy: {test_acc}') 

# Save the model 
model.save('fractal_cnn_monkeypox_model.h5')
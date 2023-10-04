from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
import os
import tensorflow as tf
from keras import layers
import time
from IPython import display
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Constants
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 3*3
NUM_CLASSES = 3
IMAGE_SIZE = 64
IMAGE_NUM_CHANNELS = 3
EPOCHS = 300
DROPOUT = 0.3
LEARNING_RATE = 0.00005
BUFFER_SIZE = 60000
BATCH_SIZE = 256
IMAGE_GENERATION_CLASS_ID = 2
TRAIN_IMAGES_FOLDER = 'images/training_simple_shapes_color'
GENERATED_IMAGES_DIR = 'generated_images_triangles_color'

FILE_ENDING = '.png'
#FILE_ENDING = '.ppm'
CLASS_NAME_ARRAY = ['1 (Square)' , '2 (Circle)', '3 (Triangle)']
#CLASS_NAME_ARRAY = ['1 (Right of way)' , '2 (Give way)', '3 (Stop)']
TRAIN_PATH_TO_FILES_ARRAY = [TRAIN_IMAGES_FOLDER+'/1/', TRAIN_IMAGES_FOLDER+'/2/', TRAIN_IMAGES_FOLDER+'/3/']

SHOW_IMAGES = False
TRAIN_MODE = False

# Trainsets
trainsets_array = []
# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Seed
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
# Cross Entropy
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Trainset class
class Trainset:
    def __init__(self, class_name, class_id, train_path_to_files):
        self.class_name = class_name
        self.class_id = class_id
        self.train_path_to_files = train_path_to_files
        self.train_file_name_set = []
        self.train_num_images = 0
        self.train_img_data = []

# Load training images
def load_train_images():
    # Create Trainset for all classes
    for c in range(0,NUM_CLASSES):
        # Create Trainset object
        trainset = Trainset(CLASS_NAME_ARRAY[c], c+1, TRAIN_PATH_TO_FILES_ARRAY[c])
        # Get all files in the current folder
        files_in_folder = os.listdir(trainset.train_path_to_files)
        # Filter out image files
        for file in files_in_folder:
            if(file.endswith(FILE_ENDING)):
                trainset.train_file_name_set.append(file)
        # Get number of images
        trainset.train_num_images = len(trainset.train_file_name_set)
        # Get the image data
        trainset.train_img_data = np.zeros((trainset.train_num_images, IMAGE_SIZE, IMAGE_SIZE, IMAGE_NUM_CHANNELS), dtype='float32')
        for i in range(0, trainset.train_num_images):
            file_name = trainset.train_file_name_set[i]
            path_to_file = trainset.train_path_to_files+file_name
            img = Image.open(path_to_file)
            img_data = asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
            img_data = img_data[:,:,0:3]
            img_data = (img_data - 127.5) / 127.5  # Normalize to [-1, 1]
            trainset.train_img_data[i, :, :, :] = img_data
        # Save trainset object
        trainsets_array.append(trainset)

# Show training images
def show_train_images():
    for i in range(0, NUM_CLASSES):
        # Print overview
        print('----------')
        print('class name: '+str(trainsets_array[i].class_name))
        print('class id: ' + str(trainsets_array[i].class_id))
        print('path: '+str(trainsets_array[i].train_path_to_files))
        print('num: '+str(trainsets_array[i].train_num_images))
        print('----------')
        # Show one image
        img_data = trainsets_array[i].train_img_data[0, :, :, :]
        img_data = (img_data*0.5 + 0.5)*255 # Normalize to [0, 255]
        img_data = img_data.astype(int)
        plt.imshow(img_data)
        plt.show()

# Generator Model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)
    return model

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Discriminator Loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator Loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Generator
generator = make_generator_model()
# Discriminator
discriminator = make_discriminator_model()

# Train Step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Generate images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(3, 3))
    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        img_data = predictions[i, :, :, :].numpy()
        img_data = (img_data * 0.5 + 0.5) * 255 # Normalize to [0, 255]
        img_data = img_data.astype(int)
        plt.imshow(img_data)
        plt.axis('off')
    plt.savefig(GENERATED_IMAGES_DIR+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

# Train function
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        # Train Steps
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)

# Main
if __name__ == '__main__':
    # Load training images
    load_train_images()
    if SHOW_IMAGES:
        # Show training images
        show_train_images()
    if TRAIN_MODE:
        # Set train images
        train_images = trainsets_array[IMAGE_GENERATION_CLASS_ID].train_img_data
        # Set train dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # Train
        train(train_dataset, EPOCHS)
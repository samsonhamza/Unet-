import h5py
from keras.callbacks import ModelCheckpoint
from preprocessing import  cleanLabelNearestNeighbour_alllabels, label012Chromosomes, makeXbyY
from Model import U_net
import numpy as np


def oneHotEncode(labels):
    '''
    One-hot encode the input labels
    '''
    num_labels = labels.shape[0]
    num_classes = 4
    labels_onehot = np.zeros((num_labels, *labels.shape[1:], num_classes))
    for label_i in range(num_labels):
        for class_i in range(num_classes):
            labels_onehot[label_i, ..., class_i] = (labels[label_i, ...] == class_i).astype(int)
    return labels_onehot


# Load the input data
with h5py.File('LowRes_13434_overlapping_pairs.h5', 'r') as f:
    x_data = f['dataset_1'][:]

# Load the labels
with h5py.File('LowRes_13434_overlapping_pairs.h5', 'r') as hf:
    labels = hf['dataset_1'][:]

# Preprocess the input data
preprocessed_images = preprocess_input(x_data)

# One-hot encode the labels
labels_onehot = oneHotEncode(labels)

# Clean incorrect labels
cleaned_labels = cleanLabelNearestNeighbour_alllabels(labels_onehot)
cleaned_labels = np.any(cleaned_labels, axis=-1).astype(int)

# Merge chromosome A and B labels
cleaned_labels = label012Chromosomes(cleaned_labels)

# Crop the data to the size required by the model
cropped_images = makeXbyY(preprocessed_images, 512, 512)
cropped_labels = makeXbyY(cleaned_labels, 512, 512)

# Split the data into training and validation sets
train_images = cropped_images[:12000]
train_labels = cropped_labels[:12000]
val_images = cropped_images[12000:]
val_labels = cropped_labels[12000:]

# Create the model
model = U_net(input_shape=(256, 256, 1))

# Define the checkpoint to save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Train the model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=8, epochs=50, callbacks=[checkpoint])

# Save the final model
model.save('final_model.h5')

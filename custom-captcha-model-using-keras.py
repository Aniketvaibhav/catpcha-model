#!/usr/bin/env python
# coding: utf-8

# # Introduction
# * This example demonstrates a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing CTC loss. For a detailed guide to layer subclassing, please check out this page in the developer guides.

# In[ ]:


import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
import keras
from keras import ops
from keras import layers


# # Load the data: Captcha Images

# * The dataset contains 400 captcha files as png images. The label for each sample is a string, the name of the file (minus the file extension). We will map each character in the string to an integer for training the model. Similary, we will need to map the predictions of the model back to strings. For this purpose we will maintain two dictionaries, mapping characters to integers, and integers to characters, respectively.

# In[ ]:


data_dir=Path("./clean_captchas")

images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))


# In[ ]:


print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)


# # Define Hyperparameters

# In[ ]:


batch_size=16
image_width=200
image_height=50

downsample_factor=4 # This reflects two MaxPooling2D((2,2)) layers
max_length=max([len(label) for label in labels])


# # Preprocessing

# In[ ]:


# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None, num_oov_indices=0, dtype="int32")

# Mapping integers back to original characters
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, num_oov_indices=0, invert=True, dtype="int32")


def split_data(images_paths_list, labels_list, train_size=0.9, shuffle=True):
    size = len(images_paths_list)
    indices = ops.arange(size)
    if shuffle:
        indices = keras.random.shuffle(indices, seed=42)
    train_samples = int(size * train_size)
    x_train_paths_arr, y_train_labels_arr = images_paths_list[indices[:train_samples]], labels_list[indices[:train_samples]]
    x_valid_paths_arr, y_valid_labels_arr = images_paths_list[indices[train_samples:]], labels_list[indices[train_samples:]]
    return x_train_paths_arr, x_valid_paths_arr, y_train_labels_arr, y_valid_labels_arr


# # Splitting data into training and validation sets

# In[ ]:


train_image_paths_np=np.array(images)
train_labels_np=np.array(labels)

x_train_paths, x_valid_paths, y_train_labels, y_valid_labels = split_data(train_image_paths_np,train_labels_np)


# In[ ]:


def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.where(img < 0.5, 0.0, 1.0)
    img = ops.image.resize(img, [image_height, image_width])
    label_tensor = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label_tensor}


# # Create Dataset objects

# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((x_train_paths, y_train_labels))
train_dataset = (train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid_paths, y_valid_labels))
validation_dataset = (validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))


# # Visualize the data

# In[ ]:


_, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
for batch in train_dataset.take(1):
    images_viz = batch["image"]
    labels_viz = batch["label"]
    for i in range(16):
        img_viz = (images_viz[i] * 255).numpy().astype("uint8")
        label_viz = tf.strings.reduce_join(num_to_char(labels_viz[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img_viz[:, :, 0], cmap="gray")
        ax[i // 4, i % 4].set_title(label_viz)
        ax[i // 4, i % 4].axis("off")
plt.show()


# # Model

# In[ ]:


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(max_num_labels_tns, current_input)

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape)
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(ops.reshape(ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),tf.reverse(label_shape, [0])))
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1]))

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)
    return tf.SparseTensor(ops.cast(indices, dtype="int64"), ops.cast(vals_sparse, dtype="int32"), ops.cast(label_shape, dtype="int64"))

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length_squeezed = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length_squeezed = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")

    sparse_labels = ctc_label_dense_to_sparse(y_true, label_length_squeezed)

    y_pred_transposed = ops.transpose(y_pred, axes=[1, 0, 2])

    loss = tf.compat.v1.nn.ctc_loss(
        labels=sparse_labels,
        inputs=y_pred_transposed,
        sequence_length=input_length_squeezed,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
        ignore_longer_outputs_than_inputs=False
    )
    return ops.expand_dims(loss, 1)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(tf.fill((batch_len, 1), max_length), dtype="int64")
        input_length_expanded = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length_expanded, label_length)
        self.add_loss(loss)
        return y_pred

def build_model():
    input_img = layers.Input(shape=(image_height, image_width, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="int32")

    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Permute((2, 1, 3))(x)
    new_shape = ((image_width // downsample_factor), (image_height // downsample_factor) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    num_classes_for_ctc = len(characters) + 1
    x = layers.Dense(num_classes_for_ctc, activation="softmax", name="dense2")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model, input_img # Return input_img as well

# Get the model
model, input_img_tensor = build_model()
model.summary(expand_nested=True)


# In[ ]:


epochs = 100
early_stopping_patience = 15

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(train_dataset,validation_data=validation_dataset,epochs=epochs,callbacks=[early_stopping, reduce_lr])


# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred_transposed = ops.transpose(y_pred, axes=[1, 0, 2])
    input_length_squeezed = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(inputs=y_pred_transposed, sequence_length=input_length_squeezed)
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(inputs=y_pred_transposed, sequence_length=input_length_squeezed, beam_width=beam_width, top_paths=top_paths)

    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

prediction_model = keras.models.Model(input_img_tensor, model.get_layer(name="dense2").output)

def decode_batch_predictions(pred):
    input_len = np.ones((pred.shape[0], 1)) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res_tensor in results:
        filtered_indices = tf.gather_nd(res_tensor, tf.where(tf.logical_and(res_tensor != -1, res_tensor < len(characters))))
        res_text = tf.strings.reduce_join(num_to_char(filtered_indices)).numpy().decode("utf-8")
        output_text.append(res_text)
    return output_text

for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label_tensor in batch_labels:
        filtered_label = tf.gather_nd(label_tensor, tf.where(label_tensor < len(characters)))
        label_text = tf.strings.reduce_join(num_to_char(filtered_label)).numpy().decode("utf-8")
        orig_texts.append(label_text)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(min(len(pred_texts), 16)):
        img_viz = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        title = f"Orig: {orig_texts[i]}\nPred: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img_viz, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()

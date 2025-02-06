#!/usr/bin/env python3
import os
import random
import itertools
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, Model, callbacks, optimizers
from sklearn.model_selection import train_test_split

import flwr as fl  # <-- Flower for Federated Learning

# ========================== CONFIG ==========================
class Config:
    # Data paths (replace with your real paths)
    IMAGE_DIR = '/home/rezaul-abedin/Developments/multimodal_fed_filckr/archive_flicker30k/flickr30k_images/flickr30k_images'
    CSV_PATH = '/home/rezaul-abedin/Developments/multimodal_fed_filckr/archive_flicker30k/flickr30k_images/results.csv'

    #IMAGE_DIR = '/home/rezaul-abedin/Developments/multimodal_deep_l/archive_flicker30k/flickr30k_images/flickr30k_images'
    #CSV_PATH = '/home/rezaul-abedin/Developments/multimodal_deep_l/archive_flicker30k/flickr30k_images/results.csv'

    MAX_ROWS = 10000   # Number of (image, caption) lines to load
    MAX_LENGTH = 30
    EMBED_DIM = 128

    NUMBER_OF_RUN = 0

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.5
    VAL_SPLIT = 0.15
    SAVE_MODEL_PATH = "multimodal_model_contrastive.h5"
    TEMPERATURE = 0.07
    IMAGE_SIZE = (128, 128)  # Smaller for quicker experimentation

    # Federated parameters
    NUM_CLIENTS = 2
    NUM_ROUNDS = 1  # How many federated rounds to run

# ========================== DATA HANDLER CLASS ==========================
class DataHandler:
    def __init__(self, csv_path, image_dir, max_rows, max_length, vocab_size=10000):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.max_rows = max_rows
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")
        self.captions_df = None
        self.images = None  # We'll store images keyed by image_name
        self.caption_dict = None  # image_name -> list of padded caption sequences

    def load_and_clean_captions(self):
        valid_rows = []
        with open(self.csv_path, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                if idx >= self.max_rows:
                    break
                fields = line.strip().split("|")
                if len(fields) == 3:
                    valid_rows.append(fields)
        self.captions_df = pd.DataFrame(valid_rows, columns=["image_name", "comment_number", "caption"])
        # Clean columns
        self.captions_df["image_name"] = (
            self.captions_df["image_name"].str.strip().str.replace('"', "", regex=False)
        )
        self.captions_df["caption"] = self.captions_df["caption"].str.strip()

    def tokenize_captions(self):
        self.tokenizer.fit_on_texts(self.captions_df["caption"])
        seqs = self.tokenizer.texts_to_sequences(self.captions_df["caption"])
        seqs_padded = pad_sequences(seqs, maxlen=self.max_length, padding="post")
        self.captions_df["caption_seq_padded"] = list(seqs_padded)

        # Store the image path for convenience
        self.captions_df["image_path"] = self.captions_df["image_name"].apply(
            lambda x: os.path.join(self.image_dir, x)
        )

    @staticmethod
    def preprocess_image(image_path, target_size):
        try:
            image = load_img(image_path, target_size=target_size)
            return img_to_array(image) / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_images_and_captions(self):
        grouped = self.captions_df.groupby("image_name")

        self.images = {}
        self.caption_dict = {}
        count = 0
        for image_name, group in grouped:
            image_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(image_path):
                img = self.preprocess_image(image_path, Config.IMAGE_SIZE)
                if img is not None:
                    self.images[image_name] = img
                    self.caption_dict[image_name] = group["caption_seq_padded"].tolist()
                    count += len(self.caption_dict[image_name])
            if count >= self.max_rows:
                break

        print(f"Loaded {len(self.images)} unique images and a total of {count} (image, caption) pairs.")

    def get_data(self):
        return self.images, self.caption_dict, len(self.tokenizer.word_index) + 1


# ========================== CUSTOM MODEL FOR CONTRASTIVE LEARNING ==========================
class ContrastiveModel(Model):
    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        # Keras will call self.metrics at the end of train_step/test_step
        return [self.loss_tracker]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        (images, captions) = data
        with tf.GradientTape() as tape:
            img_emb, txt_emb = self([images, captions], training=True)
            loss = self.contrastive_loss(img_emb, txt_emb)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (images, captions) = data
        img_emb, txt_emb = self([images, captions], training=False)
        loss = self.contrastive_loss(img_emb, txt_emb)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def contrastive_loss(self, img_emb, txt_emb):
        # Similar to CLIP-style cross-entropy loss
        logits_img_to_txt = tf.matmul(img_emb, txt_emb, transpose_b=True) / self.temperature
        logits_txt_to_img = tf.matmul(txt_emb, img_emb, transpose_b=True) / self.temperature

        batch_size = tf.shape(img_emb)[0]
        labels = tf.range(batch_size)

        loss_img = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits_img_to_txt)
        )
        loss_txt = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits_txt_to_img)
        )
        loss = (loss_img + loss_txt) / 2.0
        return loss


# ========================== MODEL BUILDER ==========================
def build_multimodal_model(vocab_size):
    """Builds the combined image-text model that outputs two embeddings."""
    # ----- IMAGE ENCODER -----
    image_input = layers.Input(shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    image_features = layers.Dense(128, activation="relu")(x)
    image_embedding = layers.Dense(256, activation=None)(image_features)
    image_embedding = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(image_embedding)

    # ----- TEXT ENCODER -----
    text_input = layers.Input(shape=(Config.MAX_LENGTH,), name="text_input")
    embedding = layers.Embedding(
        input_dim=vocab_size, output_dim=Config.EMBED_DIM, mask_zero=True
    )(text_input)
    embedding = layers.Dropout(Config.DROPOUT_RATE)(embedding)
    lstm_output = layers.LSTM(128, dropout=Config.DROPOUT_RATE)(embedding)

    text_features = layers.Dense(128, activation="relu")(lstm_output)
    text_features = layers.Dropout(Config.DROPOUT_RATE)(text_features)
    text_embedding = layers.Dense(256, activation=None)(text_features)
    text_embedding = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(text_embedding)

    # ----- Combined Model as a ContrastiveModel -----
    # We create an instance of the custom model
    contrastive_model = ContrastiveModel(
        temperature=Config.TEMPERATURE, inputs=[image_input, text_input], outputs=[image_embedding, text_embedding]
    )

    # Compile
    optimizer = optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    contrastive_model.compile(optimizer=optimizer)

    return contrastive_model


# ========================== DATA SPLITTING FOR CLIENTS ==========================
def split_data_among_clients(images_dict, captions_dict, num_clients=2, equal_samples=True):
    """
    Splits the dictionary of images and captions into `num_clients` subsets.
    If `equal_samples=True`, each client gets approximately the same number of samples.
    Returns a list of (images_dict_i, captions_dict_i) for each client i.
    """
    image_names = list(images_dict.keys())
    total_images = len(image_names)
    random.shuffle(image_names)

    # If you want equal splits:
    if equal_samples:
        chunk_size = total_images // num_clients
        subsets = []
        start = 0
        for i in range(num_clients):
            end = start + chunk_size
            if i == num_clients - 1:
                # last client gets the remainder
                subset_names = image_names[start:]
            else:
                subset_names = image_names[start:end]
            start = end

            sub_images = {k: images_dict[k] for k in subset_names}
            sub_captions = {k: captions_dict[k] for k in subset_names}
            subsets.append((sub_images, sub_captions))
        return subsets
    else:
        # Weighted/unequal splits can be done here if needed
        pass


# ========================== BATCH GENERATOR ==========================
def generate_batches(images_dict, captions_dict, batch_size, shuffle_images=True):
    """
    Yields (batch_images, batch_captions) as numpy arrays.
    """
    image_names = list(images_dict.keys())
    while True:
        if shuffle_images:
            random.shuffle(image_names)
        current_batch_images = []
        current_batch_captions = []
        for img_name in image_names:
            available_captions = captions_dict[img_name]
            chosen_caption = random.choice(available_captions)

            current_batch_images.append(images_dict[img_name])
            current_batch_captions.append(chosen_caption)

            if len(current_batch_images) == batch_size:
                yield (np.array(current_batch_images), np.array(current_batch_captions))
                current_batch_images = []
                current_batch_captions = []
        # discard remainder if < batch_size


# ========================== FLOWER CLIENT ==========================
class MultimodalClient(fl.client.NumPyClient):
    """
    Each client will:
    1) Receive the global model parameters
    2) Set those parameters locally
    3) Train on its local data for a few epochs
    4) Return updated model parameters to the server
    """

    def __init__(self, cid, model, images_dict, captions_dict, batch_size, epochs):
        self.cid = cid
        self.model = model
        self.images_dict = images_dict
        self.captions_dict = captions_dict
        self.batch_size = batch_size
        self.epochs = epochs

        # Pre-split train/val for this client
        # For demonstration, do a simple 85-15 split of its local images
        image_names = list(self.images_dict.keys())
        train_names, val_names = train_test_split(image_names, test_size=0.15, random_state=42)

        self.train_images = {k: images_dict[k] for k in train_names}
        self.train_captions = {k: captions_dict[k] for k in train_names}

        self.val_images = {k: images_dict[k] for k in val_names}
        self.val_captions = {k: captions_dict[k] for k in val_names}

        self.train_steps = max(1, len(self.train_images) // self.batch_size)
        self.val_steps = max(1, len(self.val_images) // self.batch_size)

    def get_parameters(self, config):
        """Return the current local model parameters."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train (fit) the model on local dataset."""
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Build local generators
        train_gen = generate_batches(self.train_images, self.train_captions, self.batch_size)
        val_gen = generate_batches(self.val_images, self.val_captions, self.batch_size, shuffle_images=False)

        history = self.model.fit(
            train_gen,
            steps_per_epoch=self.train_steps,
            validation_data=val_gen,
            validation_steps=self.val_steps,
            epochs=self.epochs,
            verbose=1,
        )
        # Return updated model parameters and some metrics
        return self.model.get_weights(), len(self.train_images), {
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else 0.0,
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        self.model.set_weights(parameters)
        val_gen = generate_batches(self.val_images, self.val_captions, self.batch_size, shuffle_images=False)
        result = self.model.evaluate(val_gen, steps=self.val_steps, verbose=0)
        # The first metric is "loss" because that is what we track in `ContrastiveModel`.
        loss = result[0] if isinstance(result, list) else result
        return float(loss), len(self.val_images), {"val_loss": float(loss)}


# ========================== SIMULATION RUNNER ==========================
def client_fn(cid: str):
    """
    Called by Flower for each client id 'cid'.
    We build a local model (same architecture), load data subset,
    and return a `MultimodalClient` instance.
    """
    cid_int = int(cid)

    # Build the model from scratch (same architecture).
    # In real FL you might pass the same model object, or rebuild it with the same code.
    local_model = build_multimodal_model(vocab_size=GLOBAL_VOCAB_SIZE)

    # Get the data subset for this client
    images_subset, captions_subset = CLIENT_DATA[cid_int]  # (images_dict, captions_dict)
    client = MultimodalClient(
        cid=cid,
        model=local_model,
        images_dict=images_subset,
        captions_dict=captions_subset,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
    )
    return client


if __name__ == "__main__":
    # --------------------- Load and Preprocess Data (Central Step) ---------------------
    data_handler = DataHandler(
        csv_path=Config.CSV_PATH,
        image_dir=Config.IMAGE_DIR,
        max_rows=Config.MAX_ROWS,
        max_length=Config.MAX_LENGTH,
    )
    data_handler.load_and_clean_captions()
    data_handler.tokenize_captions()
    data_handler.load_images_and_captions()
    images_dict, captions_dict, GLOBAL_VOCAB_SIZE = data_handler.get_data()

    # Split data for each client
    subsets = split_data_among_clients(
        images_dict,
        captions_dict,
        num_clients=Config.NUM_CLIENTS,
        equal_samples=True
    )
    # Make it globally accessible for client_fn
    CLIENT_DATA = {i: subsets[i] for i in range(Config.NUM_CLIENTS)}

    # --------------------- Start Flower Simulation ---------------------
    # You could customize server settings or strategy.
    # For demonstration, we simply start the simulation with default FedAvg.
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=Config.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=Config.NUM_ROUNDS),
        # You can limit concurrency if you want (e.g. num_parallel_clients=2)
    )

    print("Federated simulation finished.")

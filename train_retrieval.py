#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from base_models import MultimodalFusion, VisualizationHelper
from data_loader import FlickrDataLoader

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='contrastive_loss')
        self.temperature = temperature
    
    def call(self, y_true, embeddings):
        # Unpack image and text embeddings
        image_emb, text_emb = embeddings
        batch_size = tf.shape(image_emb)[0]
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(image_emb, text_emb, transpose_b=True)
        similarity_matrix /= self.temperature
        
        # Create labels (diagonal matrix for matching pairs)
        labels = tf.eye(batch_size, dtype=tf.float32)
        
        # Compute InfoNCE loss for image-to-text direction
        exp_sim = tf.exp(similarity_matrix)
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
        mean_log_prob_pos = tf.reduce_sum(labels * log_prob, axis=1) / tf.reduce_sum(labels, axis=1)
        loss_i2t = -mean_log_prob_pos
        
        # Compute InfoNCE loss for text-to-image direction
        log_prob = tf.transpose(similarity_matrix) - tf.math.log(tf.reduce_sum(exp_sim, axis=0, keepdims=True))
        mean_log_prob_pos = tf.reduce_sum(labels * log_prob, axis=1) / tf.reduce_sum(labels, axis=1)
        loss_t2i = -mean_log_prob_pos
        
        # Combined bidirectional loss
        loss = (loss_i2t + loss_t2i) / 2.0
        return tf.reduce_mean(loss)

class RetrievalTrainer:
    def __init__(self, model, temperature=0.07, learning_rate=1e-4):
        self.model = model
        self.loss_fn = ContrastiveLoss(temperature)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Metrics - use tf.keras.metrics.Mean()
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.viz = VisualizationHelper()  # Initialize visualization helper
    
    @tf.function
    def train_step(self, images, texts):
        with tf.GradientTape() as tape:
            # Get separate embeddings
            image_emb, text_emb = self.model((images, texts), training=True)
            # Pass dummy labels (zeros) since we don't use them
            dummy_labels = tf.zeros(tf.shape(image_emb)[0])
            loss = self.loss_fn(dummy_labels, (image_emb, text_emb))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)
        return loss
    
    def evaluate_recall_k(self, eval_data, k=[1, 5, 10]):
        # Get embeddings
        image_emb, text_emb = self.model(
            (eval_data['images'], eval_data['captions']),
            training=False
        )
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(image_emb, text_emb, transpose_b=True)
        
        recalls = {}
        for k_val in k:
            _, top_k_indices = tf.math.top_k(similarity_matrix, k=k_val)
            correct_indices = tf.range(tf.shape(similarity_matrix)[0])
            correct_in_topk = tf.reduce_any(
                tf.equal(top_k_indices, correct_indices[:, tf.newaxis]),
                axis=1
            )
            recall = tf.reduce_mean(tf.cast(correct_in_topk, tf.float32))
            recalls[k_val] = float(recall)
        
        return recalls
    
    def train(self, train_data, val_data=None, epochs=10, batch_size=32):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.train_loss_metric.reset_state()
            
            # Training loop
            num_batches = (len(train_data['images']) + batch_size - 1) // batch_size
            pbar = tqdm(range(num_batches), desc="Training")
            
            for i in pbar:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(train_data['images']))
                
                batch_imgs = tf.convert_to_tensor(train_data['images'][start_idx:end_idx], dtype=tf.float32)
                batch_texts = tf.convert_to_tensor(train_data['captions'][start_idx:end_idx], dtype=tf.int32)
                
                loss = self.train_step(batch_imgs, batch_texts)
                pbar.set_description(f"Loss: {loss:.4f}")
            
            # Collect metrics
            metrics = {
                'loss': float(self.train_loss_metric.result())
            }
            
            # Validation
            if val_data is not None:
                recalls = self.evaluate_recall_k(val_data, k=[1, 5, 10])
                metrics.update({f'recall@{k}': v for k, v in recalls.items()})
            
            # Update visualizations
            self.viz.update_history(metrics)
            self.viz.plot_training_progress()
            self.viz.log_epoch_results(epoch + 1, metrics)
            
            # Print metrics
            print("\nEpoch Results:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            
            # Visualize similarity matrix and retrieval examples every 5 epochs
            if (epoch + 1) % 5 == 0 and val_data is not None:
                # Get embeddings for visualization
                val_subset = {
                    'images': val_data['images'][:50],
                    'captions': val_data['captions'][:50]
                }
                image_emb, text_emb = self.model((val_subset['images'], val_subset['captions']), training=False)
                sim_matrix = tf.matmul(image_emb, text_emb, transpose_b=True).numpy()
                
                # Plot similarity matrix
                self.viz.plot_similarity_matrix(sim_matrix)
                
                # Visualize retrieval examples
                self.viz.visualize_retrieval_examples(self.model, val_data)

def split_and_inspect_data(processed_data, train_ratio=0.9, verbose=True, seed=42):
    """
    Shuffle and split data into train and validation sets and provide detailed inspection.
    
    Args:
        processed_data (dict): Dictionary containing 'images', 'captions', and 'vocab_size'
        train_ratio (float): Ratio of data to use for training (default: 0.9)
        verbose (bool): Whether to print inspection information
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_data, val_data) dictionaries
    """
    num_samples = len(processed_data['images'])
    
    # Create shuffled indices
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(num_samples)
    
    # Split indices
    split_idx = int(train_ratio * num_samples)
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]
    
    # Split the data using shuffled indices
    train_data = {
        'images': processed_data['images'][train_indices],
        'captions': processed_data['captions'][train_indices],
        'vocab_size': processed_data['vocab_size']
    }
    
    val_data = {
        'images': processed_data['images'][val_indices],
        'captions': processed_data['captions'][val_indices],
        'vocab_size': processed_data['vocab_size']
    }
    
    if verbose:
        print("\n=== Dataset Split Information ===")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {len(train_data['images'])} ({train_ratio*100:.1f}%)")
        print(f"Validation samples: {len(val_data['images'])} ({(1-train_ratio)*100:.1f}%)")
        print("\nShape Information:")
        print(f"Training Images shape: {train_data['images'].shape}")
        print(f"Training Captions shape: {train_data['captions'].shape}")
        print(f"Validation Images shape: {val_data['images'].shape}")
        print(f"Validation Captions shape: {val_data['captions'].shape}")
        
        # Sample inspection
        print("\n=== Sample Inspection ===")
        print("First 5 training indices:", train_indices[:5])
        print("First 5 validation indices:", val_indices[:5])
        
        # Value ranges
        print("\n=== Value Ranges ===")
        print(f"Training Images - Min: {train_data['images'].min():.2f}, Max: {train_data['images'].max():.2f}")
        print(f"Training Captions - Min: {train_data['captions'].min()}, Max: {train_data['captions'].max()}")
        
        # Memory usage
        train_mem = (train_data['images'].nbytes + train_data['captions'].nbytes) / (1024 * 1024)
        val_mem = (val_data['images'].nbytes + val_data['captions'].nbytes) / (1024 * 1024)
        print(f"\nMemory Usage:")
        print(f"Training set: {train_mem:.2f} MB")
        print(f"Validation set: {val_mem:.2f} MB")
    
    return train_data, val_data

def main():
    # Load data
    data_loader = FlickrDataLoader()
    data_loader.load_data()
    data_loader.preprocess_captions()
    data_loader.load_images()
    
    processed_data = data_loader.get_data()
    
    # Split and inspect data with shuffling
    train_data, val_data = split_and_inspect_data(
        processed_data, 
        train_ratio=0.9, 
        verbose=True,
        seed=42  # Set seed for reproducibility
    )
    
    # Create model
    fusion_model = MultimodalFusion(
        vocab_size=processed_data['vocab_size'],
        embed_dim=256,
        num_heads=8,
        num_layers=2
    )
    
    # Create trainer
    trainer = RetrievalTrainer(
        model=fusion_model,
        temperature=0.07,
        learning_rate=1e-4
    )
    
    print("\nStarting training...")
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=20,
        batch_size=32
    )

if __name__ == "__main__":
    main() 
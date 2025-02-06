#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ImageEncoder(Model):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()
        
        # CNN backbone
        self.conv_blocks = [
            self._make_conv_block(32),
            self._make_conv_block(64),
            self._make_conv_block(128)
        ]
        
        # Project patches to embedding dimension
        self.patch_proj = layers.Dense(embed_dim)
        self.patch_norm = layers.LayerNormalization()
        
    def _make_conv_block(self, filters):
        return [
            layers.Conv2D(filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2)
        ]
        
    def call(self, inputs, training=False, verbose=False):
        x = inputs
        if verbose:
            print(f"\nImage Encoder Input shape: {x.shape}")
        
        # Apply conv blocks
        for i, block in enumerate(self.conv_blocks, 1):
            for layer in block:
                x = layer(x, training=training)
            if verbose:
                print(f"After Conv Block {i}: {x.shape}")
        
        # Reshape to sequence of patches
        batch_size = tf.shape(x)[0]
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        patches = tf.reshape(x, [batch_size, h*w, -1])
        
        # Project patches to embedding dimension
        patches = self.patch_proj(patches)
        patches = self.patch_norm(patches)
        patches = tf.nn.l2_normalize(patches, axis=-1)
        
        if verbose:
            print(f"Final patch embeddings: {patches.shape}")
        return patches

class TextEncoder(Model):
    def __init__(self, vocab_size, max_length=30, embed_dim=256):
        super(TextEncoder, self).__init__()
        
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        
        # Bidirectional LSTM layers (both return sequences)
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.5)
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(embed_dim // 2, return_sequences=True, dropout=0.5)
        )
        
        self.token_proj = layers.Dense(embed_dim)
        self.token_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=False, verbose=False):
        if verbose:
            print(f"\nText Encoder Input shape: {inputs.shape}")
        
        x = self.embedding(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Project token embeddings
        x = self.token_proj(x)
        x = self.token_norm(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        
        if verbose:
            print(f"Final token embeddings: {x.shape}")
        return x

class HierarchicalCoAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross attention layers
        self.cross_attention1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.cross_attention2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        
        # Layer norms
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.norm4 = layers.LayerNormalization()
        
        # FFN layers
        self.ffn1 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.ffn2 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
    
    def call(self, image_tokens, text_tokens):
        # First cross attention: image attending to text
        attended_image = self.cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens
        )
        image_tokens = self.norm1(image_tokens + attended_image)
        image_tokens = self.norm2(image_tokens + self.ffn1(image_tokens))
        
        # Second cross attention: text attending to image
        attended_text = self.cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        text_tokens = self.norm3(text_tokens + attended_text)
        text_tokens = self.norm4(text_tokens + self.ffn2(text_tokens))
        
        return image_tokens, text_tokens

class MultimodalFusion(Model):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Image encoder
        self.image_encoder = ImageEncoder(embed_dim)
        
        # Text encoder
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        
        # Co-attention layers
        self.co_attention_layers = [
            HierarchicalCoAttention(embed_dim, num_heads) 
            for _ in range(num_layers)
        ]
        
        # Final embedding layers
        self.image_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
        
        self.text_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
    
    def call(self, inputs, training=False):
        images, texts = inputs
        
        # Get image and text tokens
        image_tokens = self.image_encoder(images, training=training)
        text_tokens = self.text_encoder(texts, training=training)
        
        # Apply co-attention layers
        for co_attn_layer in self.co_attention_layers:
            image_tokens, text_tokens = co_attn_layer(image_tokens, text_tokens)
        
        # Global pooling
        image_emb = tf.reduce_mean(image_tokens, axis=1)  # [batch_size, embed_dim]
        text_emb = tf.reduce_mean(text_tokens, axis=1)    # [batch_size, embed_dim]
        
        # Final embeddings
        image_emb = self.image_embedding(image_emb)
        text_emb = self.text_embedding(text_emb)
        
        # L2 normalization
        image_emb = tf.nn.l2_normalize(image_emb, axis=-1)
        text_emb = tf.nn.l2_normalize(text_emb, axis=-1)
        
        return image_emb, text_emb

class VisualizationHelper:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        self.training_history = {
            'loss': [],
            'recall@1': [],
            'recall@5': [],
            'recall@10': []
        }
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def update_history(self, epoch_metrics):
        """Update training history with new metrics"""
        for key, value in epoch_metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
    
    def plot_training_progress(self):
        """Plot training loss and recall metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot recalls
        plt.subplot(1, 2, 2)
        for k in [1, 5, 10]:
            key = f'recall@{k}'
            plt.plot(self.training_history[key], label=f'Recall@{k}')
        plt.title('Retrieval Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png')
        plt.close()
    
    def plot_similarity_matrix(self, similarity_matrix, k=5):
        """Plot similarity matrix heatmap with top-k matches highlighted"""
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy array if it's a tensor
        similarity_matrix = np.array(similarity_matrix)
        
        # Create mask for top-k matches
        mask = np.zeros_like(similarity_matrix, dtype=bool)  # Changed to boolean mask
        for i in range(len(similarity_matrix)):
            top_k = np.argpartition(similarity_matrix[i], -k)[-k:]
            mask[i, top_k] = True
        
        # Plot heatmap with mask
        sns.heatmap(similarity_matrix, cmap='viridis', 
                   mask=~mask,  # Using boolean mask directly
                   cbar_kws={'label': 'Similarity'})
        
        plt.title(f'Similarity Matrix (Top {k} Matches Highlighted)')
        plt.xlabel('Text Samples')
        plt.ylabel('Image Samples')
        
        plt.savefig(f'{self.save_dir}/similarity_matrix.png')
        plt.close()
    
    def visualize_retrieval_examples(self, model, val_data, num_img_queries=3, num_text_queries=2, top_k=3):
        try:
            total_samples = len(val_data['images'])
            
            # Ensure we don't request more queries than available samples
            num_img_queries = min(num_img_queries, total_samples)
            num_text_queries = min(num_text_queries, total_samples)
            
            # Get random query indices
            img_query_indices = np.random.choice(total_samples, num_img_queries, replace=False)
            text_query_indices = np.random.choice(total_samples, num_text_queries, replace=False)
            
            # Compute all embeddings in batches
            batch_size = 100
            all_image_embs = []
            all_text_embs = []
            
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                batch_images = val_data['images'][i:batch_end]
                batch_texts = val_data['captions'][i:batch_end]
                
                # Get embeddings for batch
                image_emb, text_emb = model((
                    tf.convert_to_tensor(batch_images, dtype=tf.float32),
                    tf.convert_to_tensor(batch_texts, dtype=tf.int32)
                ), training=False)
                
                all_image_embs.append(image_emb)
                all_text_embs.append(text_emb)
            
            # Concatenate all embeddings
            all_image_embs = tf.concat(all_image_embs, axis=0)
            all_text_embs = tf.concat(all_text_embs, axis=0)
            
            # Image-to-text retrieval
            print("\n=== Image-to-Text Retrieval ===")
            fig = plt.figure(figsize=(15, 4*num_img_queries))
            gs = fig.add_gridspec(num_img_queries, top_k + 1, hspace=0.4, wspace=0.3)
            
            for i, idx in enumerate(img_query_indices):
                # Get query image embedding
                query_img = val_data['images'][idx]
                query_image_emb = tf.expand_dims(all_image_embs[idx], 0)  # Add batch dimension
                
                # Compute similarities with all text embeddings
                similarities = tf.matmul(query_image_emb, all_text_embs, transpose_b=True)[0]
                top_indices = tf.argsort(similarities, direction='DESCENDING')[:top_k]
                
                # Plot query image
                ax = fig.add_subplot(gs[i, 0])
                ax.imshow(query_img)
                ax.set_title('Query Image', pad=10)
                ax.axis('off')
                
                print(f"\nQuery Image {i+1}:")
                
                # Plot matches
                for j, match_idx in enumerate(top_indices):
                    ax = fig.add_subplot(gs[i, j+1])
                    
                    match_text = f"Match {j}\nSim: {similarities[match_idx]:.2f}"
                    if match_idx == idx:
                        match_text = f"TRUE MATCH\nSim: {similarities[match_idx]:.2f}"
                        color = 'green'
                    else:
                        color = 'black'
                    
                    ax.text(0.5, 0.5, match_text, 
                           ha='center', va='center', 
                           color=color,
                           bbox=dict(facecolor='white', 
                                   edgecolor='black',
                                   boxstyle='round,pad=0.5',
                                   alpha=0.8))
                    ax.axis('off')
                    print(f"Match {j} (Sim: {similarities[match_idx]:.2f})")
            
            plt.savefig(f'{self.save_dir}/image_to_text_retrieval.png', 
                       bbox_inches='tight', dpi=300, pad_inches=0.5)
            plt.close()
            
            # Text-to-image retrieval
            print("\n=== Text-to-Image Retrieval ===")
            fig = plt.figure(figsize=(15, 4*num_text_queries))
            gs = fig.add_gridspec(num_text_queries, top_k + 1, hspace=0.4, wspace=0.3)
            
            for i, idx in enumerate(text_query_indices):
                # Get query text embedding
                query_text_emb = tf.expand_dims(all_text_embs[idx], 0)  # Add batch dimension
                
                # Compute similarities with all image embeddings
                similarities = tf.matmul(query_text_emb, all_image_embs, transpose_b=True)[0]
                top_indices = tf.argsort(similarities, direction='DESCENDING')[:top_k]
                
                # Plot query text
                ax = fig.add_subplot(gs[i, 0])
                ax.text(0.5, 0.5, "Query Text", 
                       ha='center', va='center',
                       bbox=dict(facecolor='white',
                               edgecolor='black',
                               boxstyle='round,pad=0.5',
                               alpha=0.8))
                ax.axis('off')
                
                print(f"\nQuery Text {i+1}")
                
                # Plot matches
                for j, match_idx in enumerate(top_indices):
                    ax = fig.add_subplot(gs[i, j+1])
                    ax.imshow(val_data['images'][match_idx])
                    
                    title = f"Match {j}\nSim: {similarities[match_idx]:.2f}"
                    if match_idx == idx:
                        title = f"TRUE MATCH\nSim: {similarities[match_idx]:.2f}"
                        color = 'green'
                    else:
                        color = 'black'
                    
                    ax.set_title(title, 
                                color=color, 
                                bbox=dict(facecolor='white',
                                        edgecolor='black',
                                        boxstyle='round,pad=0.5',
                                        alpha=0.8),
                                pad=10)
                    ax.axis('off')
                    print(f"Match {j} (Sim: {similarities[match_idx]:.2f})")
            
            plt.savefig(f'{self.save_dir}/text_to_image_retrieval.png', 
                       bbox_inches='tight', dpi=300, pad_inches=0.5)
            plt.close()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def log_epoch_results(self, epoch, metrics):
        """Log epoch results to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = f'{self.save_dir}/training_log.txt'
        
        with open(log_file, 'a') as f:
            f.write(f"\n=== Epoch {epoch} ({timestamp}) ===\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n") 
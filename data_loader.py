#!/usr/bin/env python3
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Fix for Qt platform plugin error
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import load_img, img_to_array
import random
import textwrap

class Config:
    # Data paths
    IMAGE_DIR = '/home/rezaul-abedin/Developments/multimodal_fed_filckr/archive_flicker30k/flickr30k_images/flickr30k_images'
    CSV_PATH = '/home/rezaul-abedin/Developments/multimodal_fed_filckr/archive_flicker30k/flickr30k_images/results.csv'
    
    # Parameters
    MAX_ROWS = 20000
    MAX_LENGTH = 30
    VOCAB_SIZE = 15000
    IMAGE_SIZE = (128, 128)

class FlickrDataLoader:
    def __init__(self):
        self.config = Config()
        self.tokenizer = Tokenizer(num_words=self.config.VOCAB_SIZE, oov_token="<unk>")
        self.data = None
        self.images = {}
        self.captions = {}
        
    def load_data(self):
        """Load and process the CSV file, keeping only the first caption for each image"""
        try:
            # Read CSV file with proper encoding and delimiter, skipping problematic whitespace
            df = pd.read_csv(
                self.config.CSV_PATH, 
                delimiter='|', 
                encoding='utf-8',
                names=['image_name', 'comment_number', 'caption'],
                skipinitialspace=True,  # Skip leading whitespace
                on_bad_lines='skip'     # Skip problematic lines
            )
            
            # Clean the columns more thoroughly
            df['image_name'] = df['image_name'].str.strip().str.replace('"', '').str.replace(' ', '')
            df['comment_number'] = pd.to_numeric(df['comment_number'].str.strip(), errors='coerce')
            df['caption'] = df['caption'].str.strip()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Filter for first comments only (comment_number = 0)
            df = df[df['comment_number'] == 0]
            
            # Limit to MAX_ROWS if specified
            if self.config.MAX_ROWS:
                df = df.head(self.config.MAX_ROWS)
            
            self.data = df[['image_name', 'caption']]
            print(f"Loaded {len(self.data)} image-caption pairs")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Let's try alternative parsing method if the first attempt fails
            try:
                print("Attempting alternative parsing method...")
                # Read the file manually to handle potential formatting issues
                rows = []
                with open(self.config.CSV_PATH, 'r', encoding='utf-8') as file:
                    for line in file:
                        # Split by '|' and clean each field
                        fields = [field.strip() for field in line.strip().split('|')]
                        if len(fields) == 3:  # Only process lines with exactly 3 fields
                            image_name = fields[0].replace('"', '').strip()
                            try:
                                comment_number = int(fields[1])
                                if comment_number == 0:  # Only keep first comments
                                    caption = fields[2].strip()
                                    rows.append([image_name, caption])
                            except ValueError:
                                continue  # Skip if comment_number is not a valid integer
                        
                        if len(rows) >= self.config.MAX_ROWS:
                            break
                            
                self.data = pd.DataFrame(rows, columns=['image_name', 'caption'])
                print(f"Successfully loaded {len(self.data)} image-caption pairs using alternative method")
                
            except Exception as e2:
                print(f"Error in alternative parsing method: {e2}")
                raise
        
    def preprocess_captions(self):
        """Tokenize and pad the captions"""
        try:
            # Fit tokenizer on captions
            self.tokenizer.fit_on_texts(self.data["caption"])
            
            # Convert captions to sequences and pad them
            sequences = self.tokenizer.texts_to_sequences(self.data["caption"])
            padded_sequences = pad_sequences(sequences, maxlen=self.config.MAX_LENGTH, padding="post")
            
            # Add processed captions to dataframe
            self.data["caption_seq"] = list(padded_sequences)
            
            # Add caption length information
            self.data["caption_length"] = self.data["caption"].str.split().str.len()
            
            # Print caption statistics
            print("\nCaption Statistics:")
            print(f"Average caption length: {self.data['caption_length'].mean():.2f} words")
            print(f"Max caption length: {self.data['caption_length'].max()} words")
            print(f"Min caption length: {self.data['caption_length'].min()} words")
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Print sample of processed captions
            print("\nSample of processed captions:")
            print(self.data[["image_name", "caption", "caption_length"]].head())
            
        except Exception as e:
            print(f"Error preprocessing captions: {e}")
            raise
            
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None
                
            image = load_img(image_path, target_size=self.config.IMAGE_SIZE)
            image_array = img_to_array(image)
            return image_array / 255.0
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
            
    def load_images(self):
        """Load all images"""
        success_count = 0
        for idx, row in self.data.iterrows():
            image_path = os.path.join(self.config.IMAGE_DIR, row["image_name"])
            img = self.preprocess_image(image_path)
            
            if img is not None:
                self.images[row["image_name"]] = img
                self.captions[row["image_name"]] = row["caption_seq"]
                success_count += 1
                
            if idx % 100 == 0:  # Progress update
                print(f"Processed {idx+1}/{len(self.data)} images...")
                
        print(f"Successfully loaded {success_count} images")
        
    def visualize_random_samples(self, num_samples=2, figsize=(15, 8)):
        """
        Visualize random sample images with their captions.
        Parameters:
            num_samples: Number of random samples to display
            figsize: Size of the figure (width, height)
        """
        try:
            if not self.images:
                raise ValueError("No images loaded. Please load images first.")
                
            # Get random image names from loaded images
            sample_images = random.sample(list(self.images.keys()), num_samples)
            
            # Create figure with consistent sizing
            plt.figure(figsize=figsize)
            
            for idx, image_name in enumerate(sample_images):
                # Get original caption
                original_caption = self.data[self.data["image_name"] == image_name]["caption"].values[0]
                
                # Load original image for visualization
                img_path = os.path.join(self.config.IMAGE_DIR, image_name)
                img = load_img(img_path)
                
                # Create subplot
                plt.subplot(1, num_samples, idx + 1)
                
                # Display image with consistent size
                plt.imshow(img)
                
                # Wrap caption text and add it as title
                wrapped_caption = '\n'.join(textwrap.wrap(original_caption, width=40))
                plt.title(f"Caption:\n{wrapped_caption}", fontsize=10, pad=10)
                
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print image details
            print("\nImage Details:")
            for image_name in sample_images:
                print(f"\nImage: {image_name}")
                print(f"Caption: {self.data[self.data['image_name'] == image_name]['caption'].values[0]}")
            
        except Exception as e:
            print(f"Error in visualization: {e}")
        
    def print_dataset_statistics(self):
        """Print comprehensive statistics about the dataset"""
        try:
            print("\n=== Dataset Statistics ===")
            print(f"\nTotal number of unique images: {len(self.images)}")
            print(f"Total number of captions: {len(self.data)}")
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Image statistics
            image_sizes = np.array([img.shape for img in self.images.values()])
            print("\nImage Statistics:")
            print(f"Image dimensions: {self.config.IMAGE_SIZE}")
            print(f"Number of color channels: {image_sizes[0][2]}")
            print(f"Total images loaded: {len(self.images)}")
            
            # Caption statistics
            print("\nCaption Statistics:")
            print("\nCaption length distribution:")
            print(self.data["caption_length"].describe())
            
            # Most common words
            print("\nTop 10 most common words:")
            word_freq = pd.Series(self.tokenizer.word_counts).sort_values(ascending=False)
            print(word_freq.head(10))
            
            # DataFrame info
            print("\nDataFrame Information:")
            print(self.data.info())
            
            # Sample of the dataset
            print("\nSample of the dataset:")
            print(self.data.head())
            
        except Exception as e:
            print(f"Error printing statistics: {e}")

    def get_data_summary(self):
        """Return a dictionary containing dataset summary"""
        return {
            "num_images": len(self.images),
            "num_captions": len(self.data),
            "vocab_size": len(self.tokenizer.word_index),
            "avg_caption_length": self.data["caption_length"].mean(),
            "max_caption_length": self.data["caption_length"].max(),
            "image_size": self.config.IMAGE_SIZE,
            "top_words": pd.Series(self.tokenizer.word_counts).sort_values(ascending=False).head(10).to_dict()
        }

    def get_data(self):
        """Return processed data for model training"""
        try:
            if not self.images or not self.captions:
                raise ValueError("Data not loaded. Call load_data(), preprocess_captions(), and load_images() first.")
            
            # Convert dictionaries to lists while maintaining alignment
            image_names = list(self.images.keys())
            images_list = [self.images[name] for name in image_names]
            captions_list = [self.captions[name] for name in image_names]
            
            return {
                'images': np.array(images_list),
                'captions': np.array(captions_list),
                'vocab_size': len(self.tokenizer.word_index) + 1
            }
            
        except Exception as e:
            print(f"Error getting data: {e}")
            raise

def main():
    try:
        # Initialize and load data
        data_loader = FlickrDataLoader()
        
        print("Loading data...")
        data_loader.load_data()
        
        print("\nPreprocessing captions...")
        data_loader.preprocess_captions()
        
        print("\nLoading images...")
        data_loader.load_images()
        
        # Print comprehensive statistics
        data_loader.print_dataset_statistics()
        
        print("\nVisualizing random samples...")
        data_loader.visualize_random_samples(num_samples=2, figsize=(15, 8))
        
        # Get data summary
        summary = data_loader.get_data_summary()
        
        print("\nDataset Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Get processed data
        processed_data = data_loader.get_data()
        print("\nProcessed Data Shapes:")
        print(f"Images shape: {processed_data['images'].shape}")
        print(f"Captions shape: {processed_data['captions'].shape}")
        print(f"Vocabulary size: {processed_data['vocab_size']}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
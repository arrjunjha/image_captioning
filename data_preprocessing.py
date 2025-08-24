import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import config

class DataPreprocessor:
    def __init__(self):
        self.captions = {}
        self.tokenizer = None
        self.max_length = 0
        
    def load_captions(self):
        """Load captions from the captions.txt file"""
        print("Loading captions...")
        
        caption_file = config.CAPTION_FILE
        
        if not os.path.exists(caption_file):
            print(f"Caption file not found: {caption_file}")
            print("Available files in data folder:")
            for file in os.listdir(config.DATA_DIR):
                print(f"  - {file}")
            return {}
        
        print(f"Using caption file: {caption_file}")
        
        # Load captions from text file
        with open(caption_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if len(line) < 2:
                    continue
                
                # Try different separators
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ',' in line:
                    parts = line.split(',', 1)
                else:
                    parts = line.split(' ', 1)
                
                if len(parts) >= 2:
                    image_name = parts[0].strip()
                    caption = parts[1].strip()
                    
                    # Remove file extension to get image_id
                    image_id = os.path.splitext(image_name)[0]
                    
                    if image_id not in self.captions:
                        self.captions[image_id] = []
                    self.captions[image_id].append(caption)
                else:
                    print(f"Skipping malformed line {line_num}: {line[:50]}...")
        
        print(f"Loaded {len(self.captions)} images with captions")
        
        # Show first few examples
        print("\nFirst 3 examples:")
        for i, (img_id, captions) in enumerate(list(self.captions.items())[:3]):
            print(f"  {img_id}: {captions[0][:60]}...")
        
        return self.captions
    
    def clean_captions(self):
        """Clean and preprocess captions"""
        print("Cleaning captions...")
        
        cleaned_captions = {}
        total_captions = 0
        
        for image_id, caption_list in self.captions.items():
            cleaned_captions[image_id] = []
            
            for caption in caption_list:
                # Convert to lowercase
                caption = caption.lower()
                
                # Remove special characters but keep spaces
                caption = re.sub(r'[^a-zA-Z\s]', '', caption)
                
                # Remove extra whitespace
                caption = re.sub(r'\s+', ' ', caption).strip()
                
                # Skip very short captions
                if len(caption.split()) < 2:
                    continue
                
                # Add start and end tokens
                caption = 'startseq ' + caption + ' endseq'
                
                cleaned_captions[image_id].append(caption)
                total_captions += 1
        
        # Remove images with no valid captions
        cleaned_captions = {k: v for k, v in cleaned_captions.items() if v}
        
        self.captions = cleaned_captions
        print(f"Cleaned {total_captions} captions for {len(cleaned_captions)} images")
        return self.captions
    
    def create_tokenizer(self):
        """Create tokenizer for captions"""
        print("Creating tokenizer...")
        
        all_captions = []
        for caption_list in self.captions.values():
            all_captions.extend(caption_list)
        
        if not all_captions:
            print("No captions found! Check your data loading.")
            return None
        
        self.tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token='<unk>')
        self.tokenizer.fit_on_texts(all_captions)
        
        # Calculate max length
        lengths = [len(caption.split()) for caption in all_captions]
        self.max_length = min(max(lengths), config.MAX_LENGTH)
        
        print(f"Total captions: {len(all_captions)}")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Max caption length: {self.max_length}")
        
        # Show some statistics
        print(f"Average caption length: {np.mean(lengths):.1f}")
        print(f"Most common words: {list(self.tokenizer.word_index.keys())[:10]}")
        
        return self.tokenizer
    
    def verify_image_caption_match(self):
        """Verify that captions have corresponding images"""
        print("\nVerifying image-caption matching...")
        
        # Get list of actual image files
        if not os.path.exists(config.IMAGE_DIR):
            print(f"Image directory not found: {config.IMAGE_DIR}")
            return
        
        image_files = [f for f in os.listdir(config.IMAGE_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        image_ids = [os.path.splitext(f)[0] for f in image_files]
        
        print(f"Found {len(image_files)} image files")
        print(f"Found {len(self.captions)} caption entries")
        
        # Find matches
        matched = 0
        for img_id in self.captions.keys():
            if img_id in image_ids:
                matched += 1
        
        print(f"Matched {matched} images with captions")
        
        if matched == 0:
            print("WARNING: No images match captions!")
            print("Sample image IDs:", image_ids[:5])
            print("Sample caption IDs:", list(self.captions.keys())[:5])
    
    def save_preprocessed_data(self):
        """Save preprocessed data"""
        if not os.path.exists(config.PROCESSED_DIR):
            os.makedirs(config.PROCESSED_DIR)
        
        # Save captions
        with open(os.path.join(config.PROCESSED_DIR, 'captions.pkl'), 'wb') as f:
            pickle.dump(self.captions, f)
        
        # Save tokenizer
        with open(os.path.join(config.PROCESSED_DIR, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save max_length
        with open(os.path.join(config.PROCESSED_DIR, 'max_length.pkl'), 'wb') as f:
            pickle.dump(self.max_length, f)
        
        print("Preprocessed data saved!")

# Run preprocessing
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    captions = preprocessor.load_captions()
    
    if captions:
        preprocessor.clean_captions()
        preprocessor.verify_image_caption_match()
        tokenizer = preprocessor.create_tokenizer()
        
        if tokenizer:
            preprocessor.save_preprocessed_data()
            print("✅ Preprocessing completed successfully!")
        else:
            print("❌ Failed to create tokenizer")
    else:
        print("❌ Failed to load captions")

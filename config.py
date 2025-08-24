import os

# Paths - Updated to match your actual structure
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")  # Changed from "image" to "Images"
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")  # Direct file path
MODEL_DIR = "models"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Model parameters
EMBEDDING_DIM = 256
LSTM_UNITS = 512
DROPOUT_RATE = 0.5
MAX_LENGTH = 34
VOCAB_SIZE = 5000

# Training parameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Image preprocessing
IMG_HEIGHT = 299
IMG_WIDTH = 299
FEATURE_DIM = 2048

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

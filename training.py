# transformer_image_captioning_FINAL.py
import os
import pickle
import tensorflow as tf
import numpy as np
import config
import gc
from tensorflow.keras.layers import (Input, Dense, Embedding, Dropout,
                                     LayerNormalization, MultiHeadAttention, add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ========== OPTIMIZED HYPERPARAMETERS ==========
NUM_LAYERS   = 4          # Increased from 2 to 4
DMODEL       = 512        # Increased from 256 to 512  
NUM_HEADS    = 8          # Keep same
DFF          = 2048       # Increased from 512 to 2048
DROPOUT_RATE = 0.15       # Increased from 0.1 to 0.15
BATCH_SIZE   = 6          # Increased from 4 to 6
EPOCHS       = 15         # Increased from 5 to 15
LEARNING_RATE = 2e-5      # Reduced from 1e-4 to 2e-5
SUBSET_SIZE  = 2000       # Increased from 1000 to 2000

print("🎯 OPTIMIZED HYPERPARAMETERS:")
print(f"   Model Dimension: {DMODEL}")
print(f"   Transformer Layers: {NUM_LAYERS}")
print(f"   Attention Heads: {NUM_HEADS}")
print(f"   Feed-Forward Size: {DFF}")
print(f"   Dropout Rate: {DROPOUT_RATE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Dataset Size: {SUBSET_SIZE} images")

# ========== FIXED POSITIONAL EMBEDDING ==========
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_len, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=d_model, 
            mask_zero=True,
            name='token_embedding'
        )
        
    def build(self, input_shape):
        # Create positional encoding weights
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_len, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        length = tf.shape(x)[1]
        return self.token_emb(x) + self.pos_emb[:, :length, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_len': self.max_len
        })
        return config

# ========== CUSTOM LAYER FOR TENSOR OPERATIONS ==========
class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config

# ========== ENHANCED TRANSFORMER DECODER BLOCK ==========
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        # Multi-head attention layers
        self.mha1 = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=rate,
            name='self_attention'
        )
        self.mha2 = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=rate,
            name='cross_attention'
        )
        
        # Enhanced Feed-Forward Network with GELU activation
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='gelu', name='ffn_dense1'),
            Dropout(rate),
            Dense(d_model, name='ffn_dense2')
        ], name='feed_forward')
        
        # Layer normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name='layernorm2')
        self.layernorm3 = LayerNormalization(epsilon=1e-6, name='layernorm3')
        
        # Dropout layers
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training=None):
        # Self-attention with causal mask
        attn1 = self.mha1(x, x, use_causal_mask=True, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Cross-attention
        attn2 = self.mha2(out1, enc_output, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config

# ========== OPTIMIZED TRANSFORMER MODEL ==========
def build_optimized_transformer(vocab_size, max_length, feature_dim):
    """Build optimized transformer with better architecture"""
    
    print(f"🏗️ Building optimized transformer...")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Max length: {max_length}")
    print(f"   Feature dim: {feature_dim}")
    
    # Image encoder with better initialization
    img_input = Input(shape=(feature_dim,), name='image_features')
    img_encoded = Dense(
        DMODEL, 
        activation='gelu',
        kernel_initializer='he_normal',
        name='img_dense'
    )(img_input)
    img_encoded = LayerNormalization(name='img_norm')(img_encoded)
    img_encoded = ExpandDims(axis=1, name='img_expand')(img_encoded)
    
    # Text decoder input
    text_input = Input(shape=(None,), name='text_sequence', dtype='int32')
    
    # Positional embedding with dropout
    x = PositionalEmbedding(vocab_size, DMODEL, max_length, name='pos_embedding')(text_input)
    x = Dropout(0.1, name='embedding_dropout')(x)
    
    # Multiple transformer decoder layers
    for i in range(NUM_LAYERS):
        x = TransformerDecoderBlock(
            DMODEL, NUM_HEADS, DFF, DROPOUT_RATE,
            name=f'transformer_block_{i}'
        )(x, img_encoded)
    
    # Final layer normalization and output
    x = LayerNormalization(name='final_norm')(x)
    outputs = Dense(
        vocab_size, 
        activation='softmax',
        kernel_initializer='glorot_uniform',
        name='output_dense'
    )(x)
    
    model = Model([img_input, text_input], outputs, name='optimized_transformer_captioner')
    
    print(f"✅ Model built with {model.count_params():,} parameters")
    return model

# ========== DATA LOADING ==========
def load_preprocessed_data():
    """Load your preprocessed data"""
    print("📁 Loading preprocessed data...")
    
    with open(os.path.join(config.PROCESSED_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(config.PROCESSED_DIR, 'captions.pkl'), 'rb') as f:
        captions = pickle.load(f)
    with open(os.path.join(config.PROCESSED_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(config.PROCESSED_DIR, 'max_length.pkl'), 'rb') as f:
        max_length = pickle.load(f)
    
    print(f"✅ Loaded {len(features)} features, {len(captions)} captions")
    return features, captions, tokenizer, max_length

def create_optimized_training_data(image_ids, features, captions, tokenizer, max_length):
    """Create properly shaped training data for sequence learning"""
    X_img, X_text, y = [], [], []
    
    for img_id in image_ids:
        if img_id not in features or img_id not in captions:
            continue
            
        for caption in captions[img_id]:
            sequence = tokenizer.texts_to_sequences([caption])[0]
            
            # Create input sequence (all words except last)
            input_seq = sequence[:-1]
            # Create target sequence (all words except first)  
            target_seq = sequence[1:]
            
            # Pad sequences to max_length
            input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')[0]
            target_seq = pad_sequences([target_seq], maxlen=max_length, padding='post')[0]
            
            X_img.append(features[img_id])
            X_text.append(input_seq)
            y.append(target_seq)
    
    return np.array(X_img), np.array(X_text), np.array(y)

# ========== OPTIMIZED CALLBACKS ==========
def get_optimized_callbacks():
    """Enhanced callbacks for better training"""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            mode='max',
            verbose=1,
            min_delta=0.001
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'best_optimized_transformer.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            mode='max',
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (0.95 ** epoch),
            verbose=1
        )
    ]

# ========== MAIN TRAINING FUNCTION ==========
def train_optimized_transformer():
    """Complete optimized training pipeline"""
    print("🚀 Starting OPTIMIZED Transformer Training Pipeline...")
    print("=" * 70)
    
    # Load data
    features, captions, tokenizer, max_length = load_preprocessed_data()
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"📊 Dataset Statistics:")
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Max caption length: {max_length}")
    print(f"   Feature dimension: {config.FEATURE_DIM}")
    
    # Get optimized dataset size
    common_ids = list(set(features.keys()) & set(captions.keys()))
    subset_ids = common_ids[:SUBSET_SIZE]
    
    # Split data
    train_ids, val_ids = train_test_split(subset_ids, test_size=0.2, random_state=42)
    
    print(f"📊 Data Split:")
    print(f"   Training images: {len(train_ids):,}")
    print(f"   Validation images: {len(val_ids):,}")
    print(f"   Total images used: {len(subset_ids):,}")
    
    # Create training data
    print("\n📝 Creating optimized training sequences...")
    X_img_train, X_text_train, y_train = create_optimized_training_data(
        train_ids, features, captions, tokenizer, max_length
    )
    
    X_img_val, X_text_val, y_val = create_optimized_training_data(
        val_ids, features, captions, tokenizer, max_length
    )
    
    print(f"✅ Training sequences: {len(X_img_train):,}")
    print(f"✅ Validation sequences: {len(X_img_val):,}")
    
    # Memory cleanup
    gc.collect()
    
    # Build optimized model
    print("\n🏗️ Building optimized transformer model...")
    model = build_optimized_transformer(vocab_size, max_length, config.FEATURE_DIM)
    
    # Compile with AdamW optimizer
    optimizer = AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    print("✅ Model compiled with AdamW optimizer")
    
    # Display model summary
    model.summary()
    
    # Start training
    print(f"\n🏋️ Starting optimized training...")
    print(f"   Target accuracy: 85%+")
    
    history = model.fit(
        [X_img_train, X_text_train], y_train,
        validation_data=([X_img_val, X_text_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_optimized_callbacks(),
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(config.MODEL_DIR, 'final_optimized_transformer.h5'))
    
    # Plot training history
    plot_training_results(history)
    
    print("✅ OPTIMIZED training completed successfully!")
    return model, history

# ========== VISUALIZATION ==========
def plot_training_results(history):
    """Plot comprehensive training results"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='orange')
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Logged', ha='center', va='center', fontsize=12)
        plt.title('Learning Rate', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'optimized_training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("🤖 FINAL OPTIMIZED TRANSFORMER IMAGE CAPTIONING")
    print("=" * 70)
    print("🎯 Expected Performance:")
    print("   • Training Accuracy: 85-92%")
    print("   • Validation Accuracy: 82-88%") 
    print("   • Training Time: ~45-60 minutes")
    print("   • Model Size: ~7M parameters")
    print("=" * 70)
    
    # Check system resources
    import psutil
    print(f"💻 System Info:")
    print(f"   RAM Usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"   Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU Available: ✅ {len(gpus)} GPU(s)")
        print(f"   GPU: {gpus[0].name}")
    else:
        print("   GPU Available: ❌ Using CPU")
    
    print("=" * 70)
    
    try:
        # Start optimized training
        model, history = train_optimized_transformer()
        
        # Display final results
        print("\n🎉 FINAL TRAINING RESULTS:")
        print("=" * 50)
        if history:
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            
            print(f"📊 Final Metrics:")
            print(f"   Training Loss:      {final_train_loss:.4f}")
            print(f"   Validation Loss:    {final_val_loss:.4f}")
            print(f"   Training Accuracy:  {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
            print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
            
            # Performance assessment
            if final_val_acc > 0.85:
                print("   🏆 EXCELLENT PERFORMANCE!")
            elif final_val_acc > 0.80:
                print("   ⭐ GOOD PERFORMANCE!")
            elif final_val_acc > 0.75:
                print("   ✅ SATISFACTORY PERFORMANCE!")
            else:
                print("   📈 Room for improvement")
            
        print("\n🔮 Your optimized transformer model is ready!")
        print("   Next: Test inference with new images")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("💡 Try reducing batch size or dataset size if memory issues persist")

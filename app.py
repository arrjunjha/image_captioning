from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import (Layer, Dense, Embedding, Dropout, 
                                     LayerNormalization, MultiHeadAttention)
from tensorflow.keras.utils import CustomObjectScope

app = Flask(__name__)

# ========== CUSTOM LAYER DEFINITIONS (Required for model loading) ==========

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

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
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
        
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='gelu', name='ffn_dense1'),
            Dropout(rate),
            Dense(d_model, name='ffn_dense2')
        ], name='feed_forward')
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name='layernorm2')
        self.layernorm3 = LayerNormalization(epsilon=1e-6, name='layernorm3')
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training=None):
        attn1 = self.mha1(x, x, use_causal_mask=True, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(out1, enc_output, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
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

# ========== GLOBAL VARIABLES ==========
model = None
tokenizer = None
max_length = None
feature_extractor = None

def load_model_components():
    """Load all model components with custom objects"""
    global model, tokenizer, max_length, feature_extractor
    
    print("Loading model components...")
    
    try:
        # Load tokenizer and max_length first
        with open('data/processed/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('data/processed/max_length.pkl', 'rb') as f:
            max_length = pickle.load(f)
        
        # Load InceptionV3 feature extractor
        base_model = InceptionV3(weights='imagenet')
        feature_extractor = KerasModel(base_model.input, base_model.layers[-2].output)
        
        # Load your trained transformer model WITH CUSTOM OBJECTS
        with CustomObjectScope({
            'PositionalEmbedding': PositionalEmbedding,
            'TransformerDecoderBlock': TransformerDecoderBlock,
            'ExpandDims': ExpandDims
        }):
            model = tf.keras.models.load_model('models/final_optimized_transformer.h5')
        
        print("✅ All components loaded successfully!")
        print(f"✅ Model has {model.count_params():,} parameters")
        
    except Exception as e:
        print(f"❌ Error loading model components: {str(e)}")
        raise e

def extract_features(image):
    """Extract features from image using InceptionV3"""
    image = image.resize((299, 299))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

def generate_caption(image_features):
    """Generate caption using transformer model"""
    start_token = tokenizer.word_index.get('startseq', 1)
    end_token = tokenizer.word_index.get('endseq', 2)
    
    sequence = [start_token]
    
    for _ in range(max_length):
        padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        
        preds = model.predict([
            image_features.reshape(1, -1), 
            padded_seq.reshape(1, -1)
        ], verbose=0)
        
        next_word = np.argmax(preds[0, len(sequence)-1, :])
        
        if next_word == end_token:
            break
            
        sequence.append(next_word)
    
    # Convert to text
    caption_words = []
    for idx in sequence[1:]:  # Skip start token
        word = tokenizer.index_word.get(idx, '')
        if word and word != 'endseq':
            caption_words.append(word)
    
    return ' '.join(caption_words).capitalize()

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and return caption"""
    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        features = extract_features(image)
        caption = generate_caption(features)
        
        return jsonify({
            'success': True,
            'caption': caption
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/video_frame', methods=['POST'])
def process_video_frame():
    """Handle video frame for live captioning"""
    try:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        features = extract_features(image)
        caption = generate_caption(features)
        
        return jsonify({
            'success': True,
            'caption': caption
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load model components on startup
    load_model_components()
    
    # Run Flask app
    print("\n🚀 Starting Flask app...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

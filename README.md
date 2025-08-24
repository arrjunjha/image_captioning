#  Transformer Image Captioning

Advanced transformer-based image captioning system with Streamlit web interface.

##  Features
- 4-layer Transformer decoder architecture
- Real-time image captioning via web app
- Live video captioning capability
- 85%+ training accuracy achieved

##  Project Structure
- `app.py` - Streamlit web application
- `config.py` - Configuration settings
- `training.py` - Model training pipeline
- `data_preproc...py` - Data preprocessing
- `feature_extra...py` - Feature extraction
- `data/` - Dataset and processed files
- `models/` - Trained model weights

##  Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset and place in `data/Images/`
4. Run preprocessing: `python data_preprocessing.py`
5. Train model: `python training.py`
6. Launch app: `streamlit run app.py`

##  Performance
- Training Accuracy: 85-90%
- Validation Accuracy: 82-88%
- Model Size: ~7M parameters
- Processing Time: 1-2 seconds per image

##  Usage
Upload images via the Streamlit interface or use live video mode for real-time captioning.

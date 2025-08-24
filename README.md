# Transformer Image Captioning

Advanced transformer-based image captioning system with **Flask web interface**.

##  Features

-  **4-layer Transformer decoder** architecture
-  **Flask web app** with HTML/CSS frontend  
-  **Photo upload + Live video** captioning
-  **82-85% validation accuracy** (15 epochs)

## 🛠️ Setup

1. **Clone repository**
git clone https://github.com/arrjunjha/image_captioning.git
cd image_captioning

text

2. **Install dependencies**
pip install -r requirements.txt

text

3. **Download trained model**

 **Download**: [Trained Model](https://drive.google.com/file/d/11RdYQTKgJ7ALcq6eD0BfjSPj9TUY1vUC/view?usp=sharing)

Place as `models/final_optimized_transformer.h5`

4. **Launch app**
python app.py

Open `http://localhost:5000`


##  Performance

- **Training Accuracy**: 92-97%
- **Validation Accuracy**: 88-93%  
- **Model Size**: ~7M parameters
- **Processing Time**: 1-2 seconds per image

##  Usage

**Photo Upload**: Upload images for instant AI captions

**Live Video**: Use webcam for real-time captioning

---

**Built with TensorFlow + Flask**

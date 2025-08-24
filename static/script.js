// Mode switching functionality
const photoModeBtn = document.getElementById('photo-mode');
const videoModeBtn = document.getElementById('video-mode');
const photoSection = document.getElementById('photo-section');
const videoSection = document.getElementById('video-section');

photoModeBtn.addEventListener('click', () => {
    photoModeBtn.classList.add('active');
    videoModeBtn.classList.remove('active');
    photoSection.style.display = 'block';
    videoSection.style.display = 'none';
    stopCamera();
});

videoModeBtn.addEventListener('click', () => {
    videoModeBtn.classList.add('active');
    photoModeBtn.classList.remove('active');
    photoSection.style.display = 'none';
    videoSection.style.display = 'block';
});

// Photo upload functionality
const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const previewArea = document.getElementById('preview-area');
const previewImage = document.getElementById('preview-image');
const loading = document.getElementById('loading');
const captionBox = document.getElementById('caption-box');
const captionText = document.getElementById('caption-text');

uploadArea.addEventListener('click', () => imageInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

async function handleImageUpload(file) {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewArea.style.display = 'flex';
        loading.style.display = 'flex';
        captionBox.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Upload to server
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        loading.style.display = 'none';
        
        if (result.success) {
            captionText.textContent = result.caption;
            captionBox.style.display = 'block';
        } else {
            alert('Error generating caption: ' + result.error);
        }
    } catch (error) {
        loading.style.display = 'none';
        alert('Error uploading image: ' + error.message);
    }
}

// Live video functionality
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
const liveCaptionText = document.getElementById('live-caption-text');

let stream = null;
let captionInterval = null;

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        
        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'inline-block';
        
        // Start captioning every 3 seconds
        captionInterval = setInterval(captureAndCaption, 3000);
        
    } catch (error) {
        alert('Error accessing camera: ' + error.message);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (captionInterval) {
        clearInterval(captionInterval);
        captionInterval = null;
    }
    
    startCameraBtn.style.display = 'inline-block';
    stopCameraBtn.style.display = 'none';
    liveCaptionText.textContent = 'Camera stopped. Click "Start Camera" to begin again.';
}

async function captureAndCaption() {
    if (!video.videoWidth) return;
    
    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await fetch('/video_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.success) {
            liveCaptionText.textContent = result.caption;
        } else {
            liveCaptionText.textContent = 'Error generating caption...';
        }
    } catch (error) {
        liveCaptionText.textContent = 'Connection error...';
    }
}

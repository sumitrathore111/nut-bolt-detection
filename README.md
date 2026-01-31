# 🔩 Nut & Bolt Detection Web Application

A real-time object detection web application that uses a YOLOv8 model to detect nuts and bolts through your webcam.

![Detection Demo](https://via.placeholder.com/800x400?text=Nut+%26+Bolt+Detection+Demo)

## 📁 Project Structure

```
nut-bolt-detection-app/
├── backend/
│   └── app.py              # Flask API server
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── styles.css          # CSS styling (dark theme)
│   └── app.js              # JavaScript logic
├── model/
│   └── best.pt             # Your trained YOLO model (add this!)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Step 1: Add Your Model

1. Download your trained model (`best.pt`) from Google Colab
2. Place it in the `model/` folder

### Step 2: Set Up Python Environment

```bash
# Navigate to project folder
cd nut-bolt-detection-app

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Start the Backend Server

```bash
# From the project root folder
python backend/app.py
```

You should see:
```
============================================================
🔩 NUT & BOLT DETECTION API SERVER
============================================================

Loading model from: .../model/best.pt
✅ Model loaded successfully!

📡 Starting server...
🌐 API will be available at: http://localhost:5000
```

### Step 4: Open the Frontend

1. Open `frontend/index.html` in your web browser
2. Click "Start Camera" to begin detection
3. Point your camera at nuts and bolts!

## 🎮 Features

- **Real-time Detection**: Live object detection through webcam
- **Modern UI**: Beautiful dark theme with responsive design
- **Statistics Dashboard**: Real-time counts and performance metrics
- **Adjustable Settings**: 
  - Confidence threshold slider
  - Detection interval control
  - Toggle labels and confidence display
- **Screenshot**: Capture and download detection results
- **Keyboard Shortcuts**:
  - `Space` - Start/Stop camera
  - `S` - Take screenshot
  - `Esc` - Close modal

## 🔧 Configuration

### Backend Configuration (in `backend/app.py`)

```python
MODEL_PATH = '../model/best.pt'  # Path to your model
CONFIDENCE_THRESHOLD = 0.25      # Minimum confidence for detections
IOU_THRESHOLD = 0.45             # IoU threshold for NMS
INPUT_SIZE = 640                 # Model input size
CLASS_NAMES = ['nut', 'bolt']    # Your class names
```

### Frontend Configuration (in `frontend/app.js`)

```javascript
const CONFIG = {
    API_URL: 'http://localhost:5000',
    DETECTION_INTERVAL: 300,     // ms between detections
    MAX_CANVAS_WIDTH: 640,       // Frame resize width
    CONFIDENCE_THRESHOLD: 0.25
};
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API and model status |
| `/detect` | POST | Run detection on image |
| `/config` | GET | Get current configuration |
| `/config` | POST | Update configuration |

### Example API Usage

```bash
# Check health
curl http://localhost:5000/health

# Run detection (with base64 image)
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

## 🛠️ Troubleshooting

### "Model not loaded" error
- Make sure `best.pt` is in the `model/` folder
- Check the model path in `backend/app.py`
- Verify you have the correct ultralytics version installed

### Camera not working
- Allow camera permissions in your browser
- Make sure no other app is using the camera
- Try a different browser (Chrome recommended)

### CORS errors
- Ensure the backend server is running
- Check that `flask-cors` is installed
- Verify the API_URL in frontend matches your server

### Slow detection
- Increase the detection interval (300-500ms recommended)
- Reduce `MAX_CANVAS_WIDTH` for smaller frame sizes
- Use a smaller/faster YOLO model

## 📋 Model Requirements

Your YOLO model should:
- Be in `.pt` format (PyTorch)
- Be trained with ultralytics/YOLOv8
- Have class names for 'nut' and 'bolt'

Update `CLASS_NAMES` in `backend/app.py` if your model uses different class names.

## 🌐 Deployment

### Local Network Access
To access from other devices on your network:
1. Find your computer's IP address
2. Update `API_URL` in frontend to use the IP
3. Run backend with `host='0.0.0.0'`

### Cloud Deployment
- **Backend**: Deploy to Heroku, Railway, or Render
- **Frontend**: Host on Netlify, Vercel, or GitHub Pages
- Update `API_URL` to your deployed backend URL

## 📄 License

This project is open source. Feel free to modify and use as needed.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Flask](https://flask.palletsprojects.com/)
- [Inter Font](https://fonts.google.com/specimen/Inter)

---

**Happy Detecting! 🔩**

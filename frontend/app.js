/**
 * Nut & Bolt Detection System - Frontend JavaScript
 * Handles webcam capture, API communication, and detection visualization
 */

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    API_URL: 'http://localhost:5000',
    DETECTION_INTERVAL: 50,  // ms between detections (50ms = ~20 FPS max)
    MAX_CANVAS_WIDTH: 640,    // Resize frames for faster processing
    CONFIDENCE_THRESHOLD: 0.25,
    COLORS: {
        nut: { stroke: '#ffa500', fill: 'rgba(255, 165, 0, 0.2)' },
        bolt: { stroke: '#00bfff', fill: 'rgba(0, 191, 255, 0.2)' },
        Nut: { stroke: '#ffa500', fill: 'rgba(255, 165, 0, 0.2)' },
        Bolt: { stroke: '#00bfff', fill: 'rgba(0, 191, 255, 0.2)' },
        default: { stroke: '#00ff00', fill: 'rgba(0, 255, 0, 0.2)' }
    }
};

// ============================================
// STATE MANAGEMENT
// ============================================
const state = {
    isRunning: false,
    stream: null,
    detectionLoop: null,
    framesProcessed: 0,
    lastFpsUpdate: Date.now(),
    frameCount: 0,
    currentFps: 0,
    showLabels: true,
    showConfidence: true
};

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    // Video
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('detectionCanvas'),
    videoOverlay: document.getElementById('videoOverlay'),

    // Buttons
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    screenshotBtn: document.getElementById('screenshotBtn'),

    // Status
    apiStatus: document.getElementById('apiStatus'),
    fpsCounter: document.getElementById('fpsCounter'),

    // Stats
    totalCount: document.getElementById('totalCount'),
    nutCount: document.getElementById('nutCount'),
    boltCount: document.getElementById('boltCount'),
    processingTime: document.getElementById('processingTime'),
    framesProcessed: document.getElementById('framesProcessed'),

    // Settings
    confidenceSlider: document.getElementById('confidenceSlider'),
    confidenceValue: document.getElementById('confidenceValue'),
    detectionInterval: document.getElementById('detectionInterval'),
    intervalValue: document.getElementById('intervalValue'),
    showLabels: document.getElementById('showLabels'),
    showConfidence: document.getElementById('showConfidence'),

    // Detections
    detectionsList: document.getElementById('detectionsList'),

    // Modal
    screenshotModal: document.getElementById('screenshotModal'),
    screenshotImage: document.getElementById('screenshotImage'),
    closeModal: document.getElementById('closeModal'),
    downloadScreenshot: document.getElementById('downloadScreenshot')
};

// Get canvas context
const ctx = elements.canvas.getContext('2d');

// ============================================
// API FUNCTIONS
// ============================================

/**
 * Check if the backend API is online
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/health`);
        const data = await response.json();

        if (data.status === 'online') {
            updateAPIStatus(true, data.model_loaded ? 'Model Ready' : 'Model Not Loaded');
            return true;
        }
    } catch (error) {
        console.error('API health check failed:', error);
        updateAPIStatus(false, 'API Offline');
    }
    return false;
}

/**
 * Send image to API for detection
 */
async function detectObjects(imageData) {
    try {
        const response = await fetch(`${CONFIG.API_URL}/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Detection API error:', error);
        return { success: false, error: error.message, detections: [] };
    }
}

/**
 * Update confidence threshold on server
 */
async function updateServerConfig() {
    try {
        await fetch(`${CONFIG.API_URL}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                confidence_threshold: CONFIG.CONFIDENCE_THRESHOLD
            })
        });
    } catch (error) {
        console.error('Config update failed:', error);
    }
}

// ============================================
// WEBCAM FUNCTIONS
// ============================================

/**
 * Start the webcam
 */
async function startCamera() {
    try {
        // Request camera access
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'  // Prefer back camera on mobile
            }
        });

        // Set video source
        elements.webcam.srcObject = state.stream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            elements.webcam.onloadedmetadata = () => {
                elements.webcam.play();
                resolve();
            };
        });

        // Setup canvas size
        setupCanvas();

        // Update UI
        elements.videoOverlay.classList.add('hidden');
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        elements.screenshotBtn.disabled = false;

        // Start detection
        state.isRunning = true;
        startDetectionLoop();

        console.log('Camera started successfully');

    } catch (error) {
        console.error('Camera error:', error);
        alert(`Failed to access camera: ${error.message}\n\nPlease ensure:\n1. Camera permissions are granted\n2. No other app is using the camera`);
    }
}

/**
 * Stop the webcam
 */
function stopCamera() {
    // Stop detection loop
    state.isRunning = false;
    if (state.detectionLoop) {
        clearInterval(state.detectionLoop);
        state.detectionLoop = null;
    }

    // Stop all tracks
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }

    // Clear canvas
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    // Update UI
    elements.videoOverlay.classList.remove('hidden');
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.screenshotBtn.disabled = true;

    // Reset stats
    updateStats({ detections: [], counts: {}, total: 0, processing_time_ms: 0 });
    elements.fpsCounter.textContent = 'FPS: --';

    console.log('Camera stopped');
}

/**
 * Setup canvas dimensions to match video
 */
function setupCanvas() {
    const video = elements.webcam;
    elements.canvas.width = video.videoWidth;
    elements.canvas.height = video.videoHeight;
}

// ============================================
// DETECTION FUNCTIONS
// ============================================

/**
 * Start the detection loop
 */
function startDetectionLoop() {
    state.detectionLoop = setInterval(async () => {
        if (!state.isRunning) return;

        // Capture frame
        const imageData = captureFrame();
        if (!imageData) return;

        // Send to API
        const result = await detectObjects(imageData);

        // Update FPS
        updateFPS();

        // Process results
        if (result.success) {
            drawDetections(result.detections);
            updateStats(result);
            updateDetectionsList(result.detections);
            state.framesProcessed++;
            elements.framesProcessed.textContent = state.framesProcessed;
        }

    }, CONFIG.DETECTION_INTERVAL);
}

/**
 * Capture current video frame as base64
 */
function captureFrame() {
    const video = elements.webcam;

    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        return null;
    }

    // Create temporary canvas for capture
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');

    // Calculate dimensions (resize for faster processing)
    let width = video.videoWidth;
    let height = video.videoHeight;

    if (width > CONFIG.MAX_CANVAS_WIDTH) {
        const ratio = CONFIG.MAX_CANVAS_WIDTH / width;
        width = CONFIG.MAX_CANVAS_WIDTH;
        height = Math.round(height * ratio);
    }

    tempCanvas.width = width;
    tempCanvas.height = height;

    // Draw video frame
    tempCtx.drawImage(video, 0, 0, width, height);

    // Convert to base64
    return tempCanvas.toDataURL('image/jpeg', 0.8);
}

/**
 * Draw detection boxes on canvas
 */
function drawDetections(detections) {
    // Clear previous drawings
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    if (!detections || detections.length === 0) return;

    // Calculate scale factor
    const video = elements.webcam;
    const scaleX = elements.canvas.width / video.videoWidth;
    const scaleY = elements.canvas.height / video.videoHeight;

    // If we resized the image for API, we need to scale back
    const apiScaleX = video.videoWidth / CONFIG.MAX_CANVAS_WIDTH;
    const apiScaleY = apiScaleX; // Maintain aspect ratio

    detections.forEach(det => {
        const { bbox, class: className, confidence } = det;

        // Scale coordinates
        const x1 = bbox.x1 * apiScaleX;
        const y1 = bbox.y1 * apiScaleX;
        const x2 = bbox.x2 * apiScaleX;
        const y2 = bbox.y2 * apiScaleX;

        const width = x2 - x1;
        const height = y2 - y1;

        // Get colors
        const colors = CONFIG.COLORS[className] || CONFIG.COLORS.default;

        // Draw filled rectangle
        ctx.fillStyle = colors.fill;
        ctx.fillRect(x1, y1, width, height);

        // Draw border
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label
        if (state.showLabels) {
            const label = state.showConfidence
                ? `${className} ${Math.round(confidence * 100)}%`
                : className;

            // Label background
            ctx.font = 'bold 16px Inter, sans-serif';
            const textWidth = ctx.measureText(label).width;
            const textHeight = 20;
            const padding = 6;

            ctx.fillStyle = colors.stroke;
            ctx.fillRect(
                x1,
                y1 - textHeight - padding,
                textWidth + padding * 2,
                textHeight + padding
            );

            // Label text
            ctx.fillStyle = '#000';
            ctx.fillText(label, x1 + padding, y1 - padding - 2);
        }
    });
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

/**
 * Update API status indicator
 */
function updateAPIStatus(isOnline, message) {
    const statusBadge = elements.apiStatus;
    const statusText = statusBadge.querySelector('.status-text');

    statusBadge.classList.remove('online', 'offline');
    statusBadge.classList.add(isOnline ? 'online' : 'offline');
    statusText.textContent = message;
}

/**
 * Update detection statistics
 */
function updateStats(result) {
    elements.totalCount.textContent = result.total || 0;
    elements.nutCount.textContent = result.counts?.nut || 0;
    elements.boltCount.textContent = result.counts?.bolt || 0;
    elements.processingTime.textContent = `${result.processing_time_ms || 0} ms`;
}

/**
 * Update detections list
 */
function updateDetectionsList(detections) {
    if (!detections || detections.length === 0) {
        elements.detectionsList.innerHTML = '<p class="no-detections">No objects detected</p>';
        return;
    }

    const html = detections.map(det => `
        <div class="detection-item">
            <span class="class-indicator ${det.class}"></span>
            <span class="class-name">${det.class}</span>
            <span class="confidence">${Math.round(det.confidence * 100)}%</span>
        </div>
    `).join('');

    elements.detectionsList.innerHTML = html;
}

/**
 * Update FPS counter
 */
function updateFPS() {
    state.frameCount++;
    const now = Date.now();
    const elapsed = now - state.lastFpsUpdate;

    if (elapsed >= 1000) {
        state.currentFps = Math.round((state.frameCount * 1000) / elapsed);
        elements.fpsCounter.textContent = `FPS: ${state.currentFps}`;
        state.frameCount = 0;
        state.lastFpsUpdate = now;
    }
}

// ============================================
// SCREENSHOT FUNCTIONS
// ============================================

/**
 * Take a screenshot with detections
 */
function takeScreenshot() {
    // Create a new canvas combining video and detections
    const screenshotCanvas = document.createElement('canvas');
    screenshotCanvas.width = elements.canvas.width;
    screenshotCanvas.height = elements.canvas.height;
    const screenshotCtx = screenshotCanvas.getContext('2d');

    // Draw video frame
    screenshotCtx.drawImage(elements.webcam, 0, 0, screenshotCanvas.width, screenshotCanvas.height);

    // Draw detection boxes
    screenshotCtx.drawImage(elements.canvas, 0, 0);

    // Add timestamp
    const timestamp = new Date().toLocaleString();
    screenshotCtx.font = '14px Inter, sans-serif';
    screenshotCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    const textWidth = screenshotCtx.measureText(timestamp).width;
    screenshotCtx.fillRect(10, screenshotCanvas.height - 30, textWidth + 20, 25);
    screenshotCtx.fillStyle = '#fff';
    screenshotCtx.fillText(timestamp, 20, screenshotCanvas.height - 12);

    // Show in modal
    const dataUrl = screenshotCanvas.toDataURL('image/png');
    elements.screenshotImage.src = dataUrl;
    elements.screenshotModal.classList.add('active');
}

/**
 * Download screenshot
 */
function downloadScreenshot() {
    const link = document.createElement('a');
    link.download = `detection_${Date.now()}.png`;
    link.href = elements.screenshotImage.src;
    link.click();
}

// ============================================
// EVENT LISTENERS
// ============================================

// Button controls
elements.startBtn.addEventListener('click', startCamera);
elements.stopBtn.addEventListener('click', stopCamera);
elements.screenshotBtn.addEventListener('click', takeScreenshot);

// Modal controls
elements.closeModal.addEventListener('click', () => {
    elements.screenshotModal.classList.remove('active');
});
elements.downloadScreenshot.addEventListener('click', downloadScreenshot);

// Close modal on background click
elements.screenshotModal.addEventListener('click', (e) => {
    if (e.target === elements.screenshotModal) {
        elements.screenshotModal.classList.remove('active');
    }
});

// Settings controls
elements.confidenceSlider.addEventListener('input', (e) => {
    CONFIG.CONFIDENCE_THRESHOLD = parseFloat(e.target.value);
    elements.confidenceValue.textContent = CONFIG.CONFIDENCE_THRESHOLD.toFixed(2);
    updateServerConfig();
});

elements.detectionInterval.addEventListener('input', (e) => {
    CONFIG.DETECTION_INTERVAL = parseInt(e.target.value);
    elements.intervalValue.textContent = CONFIG.DETECTION_INTERVAL;

    // Restart detection loop with new interval
    if (state.isRunning && state.detectionLoop) {
        clearInterval(state.detectionLoop);
        startDetectionLoop();
    }
});

elements.showLabels.addEventListener('change', (e) => {
    state.showLabels = e.target.checked;
});

elements.showConfidence.addEventListener('change', (e) => {
    state.showConfidence = e.target.checked;
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Space to start/stop
    if (e.code === 'Space' && !e.target.matches('input, textarea')) {
        e.preventDefault();
        if (state.isRunning) {
            stopCamera();
        } else {
            startCamera();
        }
    }

    // S for screenshot
    if (e.code === 'KeyS' && !e.target.matches('input, textarea') && state.isRunning) {
        e.preventDefault();
        takeScreenshot();
    }

    // Escape to close modal
    if (e.code === 'Escape') {
        elements.screenshotModal.classList.remove('active');
    }
});

// ============================================
// INITIALIZATION
// ============================================

async function init() {
    console.log('🔩 Nut & Bolt Detection System initialized');

    // Check API health
    await checkAPIHealth();

    // Periodically check API health
    setInterval(checkAPIHealth, 10000);

    // Initialize settings display
    elements.confidenceValue.textContent = CONFIG.CONFIDENCE_THRESHOLD.toFixed(2);
    elements.intervalValue.textContent = CONFIG.DETECTION_INTERVAL;
}

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

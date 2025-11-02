// ===========================
// Global Variables
// ===========================
let selectedFile = null;

// ===========================
// DOM Elements
// ===========================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const previewContent = document.getElementById('previewContent');
const removeFileBtn = document.getElementById('removeFile');
const uploadForm = document.getElementById('uploadForm');
const detectBtn = document.getElementById('detectBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const parameters = document.getElementById('parameters');
const clearBtn = document.getElementById('clearBtn');

// Parameter elements
const confidenceSlider = document.getElementById('confidence');
const iouSlider = document.getElementById('iou');
const imgSizeSelect = document.getElementById('imgSize');
const confidenceValue = document.getElementById('confidenceValue');
const iouValue = document.getElementById('iouValue');
const imgSizeValue = document.getElementById('imgSizeValue');

// ===========================
// Event Listeners
// ===========================

// Upload area click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Remove file button
removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Form submit
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await detectObjects();
});

// Clear button
clearBtn.addEventListener('click', clearResults);

// Parameter sliders
confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
});

iouSlider.addEventListener('input', (e) => {
    iouValue.textContent = parseFloat(e.target.value).toFixed(2);
});

imgSizeSelect.addEventListener('change', (e) => {
    imgSizeValue.textContent = e.target.value;
});

// ===========================
// File Handling Functions
// ===========================

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 
                         'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    
    if (!allowedTypes.includes(file.type)) {
        showNotification('Please select a valid image or video file', 'error');
        return;
    }

    // Check file size (100MB max)
    if (file.size > 100 * 1024 * 1024) {
        showNotification('File size must be less than 100MB', 'error');
        return;
    }

    selectedFile = file;
    displayFilePreview(file);
}

function displayFilePreview(file) {
    // Hide upload area, show preview
    uploadArea.style.display = 'none';
    filePreview.style.display = 'block';
    parameters.style.display = 'block';
    detectBtn.style.display = 'block';
    
    // Set file name
    fileName.textContent = file.name;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewContent.innerHTML = '';
        
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = 'Preview';
            previewContent.appendChild(img);
        } else if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = e.target.result;
            video.controls = true;
            previewContent.appendChild(video);
        }
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    filePreview.style.display = 'none';
    parameters.style.display = 'none';
    detectBtn.style.display = 'none';
    previewContent.innerHTML = '';
    fileName.textContent = '';
}

// ===========================
// Detection Functions
// ===========================

async function detectObjects() {
    if (!selectedFile) {
        showNotification('Please select a file first', 'error');
        return;
    }

    // Hide results if visible
    resultsSection.style.display = 'none';
    
    // Show loading
    detectBtn.style.display = 'none';
    loading.style.display = 'block';

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('confidence', confidenceSlider.value);
        formData.append('iou', iouSlider.value);
        formData.append('img_size', imgSizeSelect.value);

        // Send request
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Debug: Log the response data
        console.log('Detection response:', data);
        console.log('Person count in summary:', data.summary ? data.summary.person : 'No summary');
        console.log('Total objects:', data.total_objects);

        if (data.success) {
            displayResults(data);
            showNotification('Detection completed successfully!', 'success');
        } else {
            throw new Error(data.error || 'Detection failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
        detectBtn.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Display stats
    displayStats(data);
    
    // Display result image/video
    displayResultMedia(data);
    
    // Display detections table
    displayDetectionsTable(data);
}

function displayStats(data) {
    const statsContainer = document.getElementById('statsContainer');
    statsContainer.innerHTML = '';

    // Total objects stat
    const totalCard = createStatCard('fas fa-cubes', data.total_objects, 'Total Objects');
    statsContainer.appendChild(totalCard);

    // Person count (special highlight)
    const personCount = data.summary['person'] || 0;
    const personCard = createStatCard('fas fa-user', personCount, 'Persons');
    personCard.style.background = 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)';
    statsContainer.appendChild(personCard);

    // Create stat cards for other classes (excluding person since we already showed it)
    Object.entries(data.summary).forEach(([className, count]) => {
        if (className !== 'person') {
            const card = createStatCard('fas fa-tag', count, className);
            statsContainer.appendChild(card);
        }
    });

    // If no objects detected
    if (data.total_objects === 0) {
        const noObjectsCard = document.createElement('div');
        noObjectsCard.className = 'stat-card';
        noObjectsCard.innerHTML = `
            <i class="fas fa-search"></i>
            <span class="stat-value">0</span>
            <span class="stat-label">No Objects Detected</span>
        `;
        noObjectsCard.style.background = 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)';
        statsContainer.appendChild(noObjectsCard);
    }
}


function createStatCard(icon, value, label) {
    const card = document.createElement('div');
    card.className = 'stat-card';
    card.innerHTML = `
        <i class="${icon}"></i>
        <span class="stat-value">${value}</span>
        <span class="stat-label">${label}</span>
    `;
    return card;
}

function displayResultMedia(data) {
    const resultDisplay = document.getElementById('resultDisplay');
    resultDisplay.innerHTML = '';

    const mediaUrl = `/results/${data.result_file}`;

    if (data.file_type === 'image') {
        const img = document.createElement('img');
        img.src = mediaUrl;
        img.alt = 'Detection Result';
        img.style.maxWidth = '100%';
        img.style.borderRadius = '8px';
        resultDisplay.appendChild(img);
    } else {
        const videoContainer = document.createElement('div');
        videoContainer.style.width = '100%';
        videoContainer.style.maxWidth = '800px';
        videoContainer.style.margin = '0 auto';
        
        const video = document.createElement('video');
        video.src = mediaUrl;
        video.controls = true;
        video.autoplay = false;
        video.muted = true;  // Required for autoplay in some browsers
        video.style.width = '100%';
        video.style.borderRadius = '8px';
        video.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
        
        // Add error handling
        video.addEventListener('error', function(e) {
            console.error('Video error:', e);
            const errorMsg = document.createElement('p');
            errorMsg.textContent = 'Error loading video. Please try again or use a different file.';
            errorMsg.style.color = '#ef4444';
            errorMsg.style.textAlign = 'center';
            errorMsg.style.padding = '20px';
            resultDisplay.appendChild(errorMsg);
        });
        
        videoContainer.appendChild(video);
        resultDisplay.appendChild(videoContainer);
        
        // Add video info
        const info = document.createElement('p');
        info.textContent = 'Video processed successfully. Click play to view.';
        info.style.textAlign = 'center';
        info.style.color = '#6b7280';
        info.style.marginTop = '10px';
        info.style.fontSize = '14px';
        resultDisplay.appendChild(info);
    }
}

function displayDetectionsTable(data) {
    const detectionsTable = document.getElementById('detectionsTable');
    
    if (data.detections.length === 0) {
        detectionsTable.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 20px;">No objects detected in the first frame</p>';
        return;
    }

    let tableHTML = `
        <h3><i class="fas fa-list"></i> Detected Objects (First Frame - ${data.detections.length} objects)</h3>
        <div style="margin-bottom: 15px; padding: 10px; background: var(--card-bg); border-radius: 8px; border-left: 4px solid #3b82f6;">
            <i class="fas fa-info-circle" style="color: #3b82f6;"></i>
            <strong> Total across all frames: ${data.total_objects} objects</strong>
            ${data.summary['person'] ? ` • Persons: ${data.summary['person']}` : ''}
        </div>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Bounding Box</th>
                    ${data.file_type === 'video' ? '<th>Frame</th>' : ''}
                </tr>
            </thead>
            <tbody>
    `;

    data.detections.forEach((det, index) => {
        const confidence = (det.confidence * 100).toFixed(1);
        const bbox = `[${det.bbox.map(v => Math.round(v)).join(', ')}]`;
        
        // Highlight person detections
        const isPerson = det.class === 'person';
        const rowClass = isPerson ? 'class="person-row"' : '';
        
        tableHTML += `
            <tr ${rowClass}>
                <td>${index + 1}</td>
                <td><span class="detection-badge ${isPerson ? 'person-badge' : ''}">${det.class}</span></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="min-width: 50px;">${confidence}%</span>
                        <div class="confidence-bar" style="flex: 1;">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                </td>
                <td><code>${bbox}</code></td>
                ${data.file_type === 'video' ? `<td>${det.frame || 0}</td>` : ''}
            </tr>
        `;
    });

    tableHTML += `
            </tbody>
        </table>
    `;

    detectionsTable.innerHTML = tableHTML;

    // Add CSS for highlighting persons
    if (!document.getElementById('detection-styles')) {
        const style = document.createElement('style');
        style.id = 'detection-styles';
        style.textContent = `
            .person-row {
                background: rgba(59, 130, 246, 0.05) !important;
            }
            .person-row:hover {
                background: rgba(59, 130, 246, 0.1) !important;
            }
            .person-badge {
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
                color: white !important;
                font-weight: 600 !important;
            }
        `;
        document.head.appendChild(style);
    }
}

// ===========================
// Utility Functions
// ===========================

async function clearResults() {
    if (!confirm('Are you sure you want to clear all files and results?')) {
        return;
    }

    try {
        const response = await fetch('/clear', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            // Reset UI
            resetUpload();
            resultsSection.style.display = 'none';
            showNotification('Files cleared successfully', 'success');
        } else {
            throw new Error(data.error || 'Failed to clear files');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;

    // Add styles if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 12px;
                color: white;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
                z-index: 9999;
                animation: slideIn 0.3s ease;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                max-width: 400px;
            }
            .notification-success {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            .notification-error {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }
            .notification-info {
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            }
            @keyframes slideIn {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Add to body
    document.body.appendChild(notification);

    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 4000);
}

// ===========================
// Initialize
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    console.log('YOLOv5 Web Interface Loaded');
    
    // Check server health
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (data.model_loaded) {
                console.log('✓ Model loaded successfully');
                console.log('✓ Device:', data.device);
            } else {
                showNotification('Warning: Model not loaded', 'error');
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
            showNotification('Server connection error', 'error');
        });
});

// ===========================
// Camera Detection Functions
// ===========================

let cameraActive = false;
let cameraStream = null;

// DOM Elements for Camera
const cameraSection = document.getElementById('cameraSection');
const cameraToggle = document.getElementById('cameraToggle');
const cameraFeed = document.getElementById('cameraFeed');
const cameraStatus = document.getElementById('cameraStatus');
const takeSnapshotBtn = document.getElementById('takeSnapshot');
const cameraLoading = document.getElementById('cameraLoading');

// Event Listeners for Camera
cameraToggle.addEventListener('click', toggleCamera);
takeSnapshotBtn.addEventListener('click', takeSnapshot);

async function toggleCamera() {
    if (cameraActive) {
        await stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        cameraLoading.style.display = 'block';
        cameraToggle.disabled = true;
        
        const response = await fetch('/camera/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            cameraActive = true;
            cameraToggle.innerHTML = '<i class="fas fa-video-slash"></i> Stop Camera';
            cameraToggle.classList.remove('btn-primary');
            cameraToggle.classList.add('btn-danger');
            cameraStatus.textContent = 'Camera Active - Live Detection Running';
            cameraStatus.className = 'status-active';
            
            // Start showing camera feed
            cameraFeed.src = '/camera_feed?' + new Date().getTime(); // Cache buster
            cameraFeed.style.display = 'block';
            takeSnapshotBtn.style.display = 'block';
            
            showNotification('Camera started successfully', 'success');
        } else {
            throw new Error(data.error || 'Failed to start camera');
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        cameraLoading.style.display = 'none';
        cameraToggle.disabled = false;
    }
}

async function stopCamera() {
    try {
        cameraToggle.disabled = true;
        
        const response = await fetch('/camera/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            cameraActive = false;
            cameraToggle.innerHTML = '<i class="fas fa-video"></i> Start Camera';
            cameraToggle.classList.remove('btn-danger');
            cameraToggle.classList.add('btn-primary');
            cameraStatus.textContent = 'Camera Inactive';
            cameraStatus.className = 'status-inactive';
            
            // Stop camera feed
            cameraFeed.src = '';
            cameraFeed.style.display = 'none';
            takeSnapshotBtn.style.display = 'none';
            
            showNotification('Camera stopped successfully', 'success');
        } else {
            throw new Error(data.error || 'Failed to stop camera');
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        cameraToggle.disabled = false;
    }
}

async function takeSnapshot() {
    if (!cameraActive) {
        showNotification('Please start the camera first', 'error');
        return;
    }
    
    try {
        takeSnapshotBtn.disabled = true;
        takeSnapshotBtn.innerHTML = '<i class="fas fa-camera"></i> Taking Snapshot...';
        
        const response = await fetch('/camera/snapshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Snapshot saved successfully!', 'success');
            
            // Optionally display the snapshot in results section
            displayCameraSnapshot(data.snapshot_file);
        } else {
            throw new Error(data.error || 'Failed to take snapshot');
        }
    } catch (error) {
        console.error('Error taking snapshot:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        takeSnapshotBtn.disabled = false;
        takeSnapshotBtn.innerHTML = '<i class="fas fa-camera"></i> Take Snapshot';
    }
}

function displayCameraSnapshot(snapshotFile) {
    // Create a temporary display for the snapshot
    const snapshotDisplay = document.createElement('div');
    snapshotDisplay.className = 'snapshot-preview';
    snapshotDisplay.innerHTML = `
        <div style="text-align: center; margin: 20px 0;">
            <h4><i class="fas fa-camera"></i> Camera Snapshot</h4>
            <img src="/results/${snapshotFile}" alt="Camera Snapshot" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="margin-top: 10px;">
                <a href="/results/${snapshotFile}" download="${snapshotFile}" class="btn btn-secondary">
                    <i class="fas fa-download"></i> Download
                </a>
            </div>
        </div>
    `;
    
    // Insert after camera section or in results section
    cameraSection.appendChild(snapshotDisplay);
    
    // Auto-remove after 30 seconds
    setTimeout(() => {
        if (snapshotDisplay.parentNode) {
            snapshotDisplay.remove();
        }
    }, 30000);
}

// Check camera status on page load
async function checkCameraStatus() {
    try {
        const response = await fetch('/camera/status');
        const data = await response.json();
        
        if (data.active) {
            // Camera was active, restart it
            await startCamera();
        }
    } catch (error) {
        console.error('Error checking camera status:', error);
    }
}
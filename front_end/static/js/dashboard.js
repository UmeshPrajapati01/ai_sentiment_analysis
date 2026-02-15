/**
 * ==========================================
 * CAT EMOTION DETECTION - DASHBOARD INTERACTIVITY
 * Premium JavaScript with Drag & Drop, Animations, and Visual Effects
 * ==========================================
 */

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', function () {
    initDragAndDrop();
    initFilePreview();
    initFormSubmission();
    initAnimations();
    initEmotionIcons();
});

// ==========================================
// DRAG AND DROP FILE UPLOAD
// ==========================================
function initDragAndDrop() {
    const fileInput = document.getElementById('file');
    const uploadForm = document.getElementById('upload-form');

    if (!fileInput || !uploadForm) return;

    // Create drop zone overlay
    const dropZone = document.createElement('div');
    dropZone.className = 'drop-zone-overlay';
    dropZone.innerHTML = `
        <div class="drop-zone-content">
            <i class="bi bi-cloud-arrow-up-fill" style="font-size: 4rem; color: var(--primary-color);"></i>
            <h3 class="mt-3">Drop your file here</h3>
            <p class="text-muted">Image or Audio file</p>
        </div>
    `;
    dropZone.style.display = 'none';
    document.body.appendChild(dropZone);

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
        uploadForm.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        document.body.addEventListener(eventName, () => {
            dropZone.style.display = 'flex';
            dropZone.classList.add('active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, () => {
            dropZone.style.display = 'none';
            dropZone.classList.remove('active');
        }, false);
    });

    // Handle dropped files
    document.body.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: fileInput });

            // Add success animation
            showNotification('File added successfully!', 'success');
        }
    }
}

// ==========================================
// FILE PREVIEW
// ==========================================
function initFilePreview() {
    const fileInput = document.getElementById('file');
    if (!fileInput) return;

    fileInput.addEventListener('change', handleFileSelect);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Remove existing preview
    const existingPreview = document.querySelector('.file-preview-container');
    if (existingPreview) {
        existingPreview.remove();
    }

    // Create preview container
    const previewContainer = document.createElement('div');
    previewContainer.className = 'file-preview-container';

    const fileType = file.type.split('/')[0];
    const fileName = file.name;
    const fileSize = formatFileSize(file.size);

    if (fileType === 'image') {
        // Image preview
        const reader = new FileReader();
        reader.onload = function (e) {
            previewContainer.innerHTML = `
                <div class="preview-card">
                    <div class="preview-header">
                        <span class="badge bg-info">
                            <i class="bi bi-image"></i> Image Preview
                        </span>
                        <button type="button" class="btn-close-preview" onclick="removePreview()">
                            <i class="bi bi-x-lg"></i>
                        </button>
                    </div>
                    <div class="preview-body">
                        <img src="${e.target.result}" alt="Preview" class="preview-image">
                        <div class="preview-info">
                            <p class="file-name"><i class="bi bi-file-earmark-image"></i> ${fileName}</p>
                            <p class="file-size">${fileSize}</p>
                        </div>
                    </div>
                </div>
            `;
            insertPreview(previewContainer);
        };
        reader.readAsDataURL(file);
    } else if (fileType === 'audio') {
        // Audio preview
        const audioURL = URL.createObjectURL(file);
        previewContainer.innerHTML = `
            <div class="preview-card">
                <div class="preview-header">
                    <span class="badge bg-warning">
                        <i class="bi bi-music-note-beamed"></i> Audio Preview
                    </span>
                    <button type="button" class="btn-close-preview" onclick="removePreview()">
                        <i class="bi bi-x-lg"></i>
                    </button>
                </div>
                <div class="preview-body">
                    <div class="audio-player-container">
                        <i class="bi bi-disc" style="font-size: 4rem; color: var(--warning-gradient);"></i>
                        <audio controls class="audio-player">
                            <source src="${audioURL}" type="${file.type}">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                    <div class="preview-info">
                        <p class="file-name"><i class="bi bi-file-earmark-music"></i> ${fileName}</p>
                        <p class="file-size">${fileSize}</p>
                    </div>
                </div>
            </div>
        `;
        insertPreview(previewContainer);
    }
}

function insertPreview(previewContainer) {
    const fileInputParent = document.getElementById('file').parentElement;
    fileInputParent.parentElement.insertBefore(previewContainer, fileInputParent.nextSibling);

    // Animate preview entrance
    setTimeout(() => {
        previewContainer.classList.add('show');
    }, 10);
}

function removePreview() {
    const preview = document.querySelector('.file-preview-container');
    if (preview) {
        preview.classList.remove('show');
        setTimeout(() => preview.remove(), 300);
    }
    document.getElementById('file').value = '';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==========================================
// FORM SUBMISSION WITH LOADING STATE
// ==========================================
function initFormSubmission() {
    const form = document.getElementById('upload-form');
    if (!form) return;

    form.addEventListener('submit', function (e) {
        // Show loading overlay
        showLoadingOverlay();

        // Disable submit button
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
        }
    });
}

function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner-large"></div>
            <h3 class="mt-4">Analyzing Cat Emotion...</h3>
            <p class="text-muted">This may take a few seconds</p>
            <div class="loading-bar">
                <div class="loading-bar-progress"></div>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    setTimeout(() => {
        overlay.classList.add('show');
    }, 10);
}

// ==========================================
// ANIMATIONS
// ==========================================
function initAnimations() {
    // Animate cards on page load
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });

    // Add parallax effect to background
    document.addEventListener('mousemove', (e) => {
        const moveX = (e.clientX - window.innerWidth / 2) * 0.01;
        const moveY = (e.clientY - window.innerHeight / 2) * 0.01;

        document.body.style.backgroundPosition = `${50 + moveX}% ${50 + moveY}%`;
    });
}

// ==========================================
// EMOTION-SPECIFIC ICONS AND CELEBRATIONS
// ==========================================
function initEmotionIcons() {
    const predictionElement = document.querySelector('.display-3');
    if (!predictionElement) return;

    const emotion = predictionElement.textContent.trim().toLowerCase();
    const emotionConfig = getEmotionConfig(emotion);

    if (emotionConfig) {
        // Add emotion icon
        const iconElement = document.createElement('div');
        iconElement.className = 'emotion-icon-large';
        iconElement.innerHTML = emotionConfig.icon;
        iconElement.style.fontSize = '5rem';

        predictionElement.parentElement.insertBefore(iconElement, predictionElement);

        // Trigger celebration effect
        if (emotionConfig.celebrate) {
            triggerConfetti();
        }

        // Add emotion-specific styling
        const resultContainer = predictionElement.closest('.mt-5');
        if (resultContainer) {
            resultContainer.style.background = emotionConfig.gradient;
        }
    }
}

function getEmotionConfig(emotion) {
    const configs = {
        'happy': {
            icon: '😸',
            gradient: 'linear-gradient(135deg, rgba(255, 234, 167, 0.9) 0%, rgba(255, 209, 102, 0.9) 100%)',
            celebrate: true
        },
        'sad': {
            icon: '😿',
            gradient: 'linear-gradient(135deg, rgba(189, 195, 199, 0.9) 0%, rgba(149, 165, 166, 0.9) 100%)',
            celebrate: false
        },
        'angry': {
            icon: '😾',
            gradient: 'linear-gradient(135deg, rgba(255, 118, 117, 0.9) 0%, rgba(253, 121, 168, 0.9) 100%)',
            celebrate: false
        },
        'defense': {
            icon: '🙀',
            gradient: 'linear-gradient(135deg, rgba(252, 92, 125, 0.9) 0%, rgba(106, 130, 251, 0.9) 100%)',
            celebrate: false
        },
        'fighting': {
            icon: '😼',
            gradient: 'linear-gradient(135deg, rgba(235, 87, 87, 0.9) 0%, rgba(219, 10, 91, 0.9) 100%)',
            celebrate: false
        },
        'resting': {
            icon: '😴',
            gradient: 'linear-gradient(135deg, rgba(162, 155, 254, 0.9) 0%, rgba(199, 146, 234, 0.9) 100%)',
            celebrate: true
        },
        'mating': {
            icon: '😻',
            gradient: 'linear-gradient(135deg, rgba(253, 121, 168, 0.9) 0%, rgba(255, 77, 109, 0.9) 100%)',
            celebrate: false
        },
        'warning': {
            icon: '⚠️',
            gradient: 'linear-gradient(135deg, rgba(250, 177, 160, 0.9) 0%, rgba(255, 107, 107, 0.9) 100%)',
            celebrate: false
        },
        'paining': {
            icon: '😢',
            gradient: 'linear-gradient(135deg, rgba(108, 92, 231, 0.9) 0%, rgba(97, 162, 240, 0.9) 100%)',
            celebrate: false
        },
        'huntingmind': {
            icon: '🎯',
            gradient: 'linear-gradient(135deg, rgba(74, 144, 226, 0.9) 0%, rgba(0, 242, 254, 0.9) 100%)',
            celebrate: false
        },
        'motherCall': {
            icon: '💕',
            gradient: 'linear-gradient(135deg, rgba(255, 159, 243, 0.9) 0%, rgba(179, 136, 255, 0.9) 100%)',
            celebrate: true
        }
    };

    return configs[emotion] || null;
}

// ==========================================
// CONFETTI EFFECT
// ==========================================
function triggerConfetti() {
    const duration = 3 * 1000;
    const animationEnd = Date.now() + duration;
    const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 9999 };

    function randomInRange(min, max) {
        return Math.random() * (max - min) + min;
    }

    const interval = setInterval(function () {
        const timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
            return clearInterval(interval);
        }

        const particleCount = 50 * (timeLeft / duration);

        // Create confetti particles
        for (let i = 0; i < particleCount; i++) {
            createConfettiParticle(
                randomInRange(0.1, 0.3),
                randomInRange(0.1, 0.3)
            );
        }
    }, 250);
}

function createConfettiParticle(x, y) {
    const particle = document.createElement('div');
    particle.className = 'confetti-particle';

    const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
    particle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
    particle.style.left = `${x * 100}%`;
    particle.style.top = `${y * 100}%`;
    particle.style.width = `${Math.random() * 10 + 5}px`;
    particle.style.height = particle.style.width;

    document.body.appendChild(particle);

    setTimeout(() => {
        particle.remove();
    }, 3000);
}

// ==========================================
// NOTIFICATIONS
// ==========================================
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="bi bi-check-circle-fill me-2"></i>
        ${message}
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Auto-detect file type and select radio button
document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file');
    if (!fileInput) return;

    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (!file) return;

        const fileType = file.type.split('/')[0];

        if (fileType === 'image') {
            document.getElementById('type_image').checked = true;
        } else if (fileType === 'audio') {
            document.getElementById('type_audio').checked = true;
        }
    });
});

// Smooth scroll to results
window.addEventListener('load', function () {
    const predictionElement = document.querySelector('.display-3');
    if (predictionElement) {
        setTimeout(() => {
            predictionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 500);
    }
});

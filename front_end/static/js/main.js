// ==========================================
// CAT EMOTION DETECTION - MODERN JAVASCRIPT
// ==========================================

console.log("🐱 Cat Emotion Detector Loaded Successfully!");

// ==========================================
// SMOOTH SCROLL FOR NAVIGATION
// ==========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ==========================================
// FILE INPUT ENHANCEMENT
// ==========================================
const fileInputs = document.querySelectorAll('input[type="file"]');

fileInputs.forEach(input => {
    input.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            // Show file name
            const fileName = file.name;
            console.log(`📁 File selected: ${fileName}`);

            // Create file preview if it's an image
            if (file.type.startsWith('image/')) {
                createImagePreview(file, input);
            }

            // Add feedback animation
            input.style.borderColor = '#667eea';
            input.style.background = 'rgba(102, 126, 234, 0.1)';

            setTimeout(() => {
                input.style.borderColor = '';
                input.style.background = '';
            }, 2000);
        }
    });
});

// Create image preview
function createImagePreview(file, inputElement) {
    const reader = new FileReader();
    reader.onload = function (e) {
        // Check if preview already exists
        let preview = inputElement.parentElement.querySelector('.file-preview');

        if (!preview) {
            preview = document.createElement('div');
            preview.className = 'file-preview mt-3';
            preview.style.cssText = `
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
                max-width: 300px;
                margin: 0 auto;
                animation: fadeInUp 0.4s ease;
            `;
            inputElement.parentElement.appendChild(preview);
        }

        preview.innerHTML = `
            <img src="${e.target.result}" 
                 alt="Preview" 
                 style="width: 100%; height: auto; display: block;">
            <div style="padding: 0.75rem; text-align: center; background: rgba(102, 126, 234, 0.95); color: white; font-weight: 600;">
                📸 ${file.name}
            </div>
        `;
    };
    reader.readAsDataURL(file);
}

// ==========================================
// FORM SUBMISSION LOADING STATE
// ==========================================
const forms = document.querySelectorAll('form');

forms.forEach(form => {
    form.addEventListener('submit', function (e) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && !submitBtn.classList.contains('no-loading')) {
            // Store original text
            const originalText = submitBtn.innerHTML;

            // Add loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = `
                <span class="loading-spinner" style="
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border: 2px solid rgba(255,255,255,0.3);
                    border-radius: 50%;
                    border-top-color: white;
                    animation: spin 0.8s linear infinite;
                    margin-right: 8px;
                    vertical-align: middle;
                "></span>
                Processing...
            `;

            // Add spin animation if not exists
            if (!document.querySelector('#spin-keyframes')) {
                const style = document.createElement('style');
                style.id = 'spin-keyframes';
                style.innerHTML = `
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                `;
                document.head.appendChild(style);
            }
        }
    });
});

// ==========================================
// ALERT AUTO-DISMISS
// ==========================================
const alerts = document.querySelectorAll('.alert');

alerts.forEach(alert => {
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.style.animation = 'slideUp 0.4s ease';
        setTimeout(() => {
            alert.remove();
        }, 400);
    }, 5000);

    // Add close button if not exists
    if (!alert.querySelector('.btn-close')) {
        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'btn-close';
        closeBtn.setAttribute('data-bs-dismiss', 'alert');
        closeBtn.style.cssText = 'position: absolute; right: 1rem; top: 1rem;';
        alert.style.position = 'relative';
        alert.appendChild(closeBtn);
    }
});

// Add slideUp animation
if (!document.querySelector('#slide-keyframes')) {
    const style = document.createElement('style');
    style.id = 'slide-keyframes';
    style.innerHTML = `
        @keyframes slideUp {
            from {
                opacity: 1;
                transform: translateY(0);
            }
            to {
                opacity: 0;
                transform: translateY(-20px);
            }
        }
    `;
    document.head.appendChild(style);
}

// ==========================================
// NAVBAR SCROLL EFFECT
// ==========================================
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

if (navbar) {
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.25)';
            navbar.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.2)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.15)';
            navbar.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
        }

        lastScroll = currentScroll;
    });
}

// ==========================================
// CARD TILT EFFECT (3D HOVER)
// ==========================================
const cards = document.querySelectorAll('.card');

cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = (y - centerY) / 20;
        const rotateY = (centerX - x) / 20;

        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
    });

    card.addEventListener('mouseleave', () => {
        card.style.transform = '';
    });
});

// ==========================================
// RESULT ANIMATION
// ==========================================
const resultElements = document.querySelectorAll('.display-4');

resultElements.forEach(element => {
    // Animate result text
    element.style.opacity = '0';
    element.style.transform = 'scale(0.8)';

    setTimeout(() => {
        element.style.transition = 'all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        element.style.opacity = '1';
        element.style.transform = 'scale(1)';
    }, 100);
});

// ==========================================
// TABLE ROW CLICK EFFECT
// ==========================================
const tableRows = document.querySelectorAll('.table tbody tr');

tableRows.forEach(row => {
    row.style.cursor = 'pointer';

    row.addEventListener('click', function () {
        // Add ripple effect
        const ripple = document.createElement('div');
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(102, 126, 234, 0.3);
            width: 20px;
            height: 20px;
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;

        this.style.position = 'relative';
        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
if (!document.querySelector('#ripple-keyframes')) {
    const style = document.createElement('style');
    style.id = 'ripple-keyframes';
    style.innerHTML = `
        @keyframes ripple {
            to {
                transform: scale(20);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// ==========================================
// BUTTON CLICK ANIMATION
// ==========================================
const buttons = document.querySelectorAll('.btn');

buttons.forEach(button => {
    button.addEventListener('click', function (e) {
        const x = e.clientX - this.getBoundingClientRect().left;
        const y = e.clientY - this.getBoundingClientRect().top;

        const ripple = document.createElement('span');
        ripple.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.6);
            transform: translate(-50%, -50%);
            animation: rippleClick 0.6s ease-out;
        `;

        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});

// Add click ripple animation
if (!document.querySelector('#ripple-click-keyframes')) {
    const style = document.createElement('style');
    style.id = 'ripple-click-keyframes';
    style.innerHTML = `
        @keyframes rippleClick {
            to {
                width: 300px;
                height: 300px;
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// ==========================================
// PAGE LOAD ANIMATION
// ==========================================
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';

    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// ==========================================
// CONSOLE EASTER EGG
// ==========================================
console.log(`
%c🐱 Cat Emotion Detection System 
%cBuilt with ❤️ using Flask & PyTorch
%c⚡ All systems operational!
`,
    'color: #667eea; font-size: 20px; font-weight: bold;',
    'color: #764ba2; font-size: 14px;',
    'color: #4facfe; font-size: 12px;'
);

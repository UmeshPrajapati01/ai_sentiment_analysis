/* ===== MewConnect - Main JavaScript ===== */
/* Integrated with Flask Backend */

// ===== NAVBAR SCROLL EFFECT =====
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  // Mobile toggle
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
      toggle.classList.toggle('active');
    });
  }
}

// ===== HERO PARTICLES =====
function initParticles() {
  const container = document.querySelector('.hero-particles');
  if (!container) return;

  for (let i = 0; i < 30; i++) {
    const particle = document.createElement('div');
    particle.classList.add('particle');
    const size = Math.random() * 6 + 3;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    particle.style.animationDelay = `${Math.random() * 5}s`;
    particle.style.animationDuration = `${Math.random() * 3 + 2}s`;
    container.appendChild(particle);
  }
}

// ===== FLOATING PAW PRINTS (Landing Page) =====
function initPawPrints() {
  const hero = document.querySelector('.hero');
  if (!hero) return;

  setInterval(() => {
    const paw = document.createElement('span');
    paw.textContent = '🐾';
    paw.style.cssText = `
      position: absolute;
      font-size: ${Math.random() * 20 + 14}px;
      left: ${Math.random() * 100}%;
      top: ${Math.random() * 100}%;
      pointer-events: none;
      opacity: 0;
      z-index: 1;
      animation: pawPrint 3s ease forwards;
    `;
    hero.appendChild(paw);
    setTimeout(() => paw.remove(), 3000);
  }, 2000);
}

// ===== SCROLL ANIMATIONS (Intersection Observer) =====
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('anim-fadeInUp');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    observer.observe(el);
  });
}

// ===== FORM VALIDATION =====
function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validatePassword(password) {
  return password.length >= 6;
}

function showFieldError(input, message) {
  input.style.borderColor = '#ef4444';
  let errorEl = input.parentElement.querySelector('.field-error');
  if (!errorEl) {
    errorEl = document.createElement('span');
    errorEl.className = 'field-error';
    errorEl.style.cssText = 'color: #ef4444; font-size: 0.78rem; margin-top: 4px; display: block;';
    input.parentElement.appendChild(errorEl);
  }
  errorEl.textContent = message;
}

function clearFieldError(input) {
  input.style.borderColor = '#e2e8f0';
  const errorEl = input.parentElement.querySelector('.field-error');
  if (errorEl) errorEl.remove();
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'success') {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);
  }

  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.innerHTML = `
    <span style="font-size:1.3rem">${type === 'success' ? '✅' : '❌'}</span>
    <span>${message}</span>
  `;

  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}

// ===== REGISTER FORM - Client-side validation before submit =====
function initRegisterForm() {
  const form = document.getElementById('registerForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    let isValid = true;

    const fullName = form.querySelector('#fullName');
    const email = form.querySelector('#email');
    const password = form.querySelector('#password');
    const confirmPassword = form.querySelector('#confirm_password');

    // Clear previous errors
    [fullName, email, password].forEach(clearFieldError);
    if (confirmPassword) clearFieldError(confirmPassword);

    if (fullName && !fullName.value.trim()) {
      showFieldError(fullName, 'Full name is required');
      isValid = false;
    }
    if (!validateEmail(email.value)) {
      showFieldError(email, 'Please enter a valid Gmail address');
      isValid = false;
    }
    if (!validatePassword(password.value)) {
      showFieldError(password, 'Password must be at least 6 characters');
      isValid = false;
    }
    if (confirmPassword && password.value !== confirmPassword.value) {
      showFieldError(confirmPassword, 'Passwords do not match');
      isValid = false;
    }

    if (!isValid) {
      e.preventDefault(); // Stop form submission if validation fails
    }
    // If valid, form submits normally to Flask backend via POST
  });
}

// ===== LOGIN FORM - Client-side validation before submit =====
function initLoginForm() {
  const form = document.getElementById('loginForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    let isValid = true;

    const email = form.querySelector('#loginEmail');
    const password = form.querySelector('#loginPassword');

    [email, password].forEach(clearFieldError);

    if (!validateEmail(email.value)) {
      showFieldError(email, 'Please enter a valid email');
      isValid = false;
    }
    if (!password.value) {
      showFieldError(password, 'Password is required');
      isValid = false;
    }

    if (!isValid) {
      e.preventDefault(); // Stop form submission if validation fails
    }
    // If valid, form submits normally to Flask backend via POST
  });
}

// ===== DASHBOARD INIT =====
function initDashboard() {
  // Init chart bars animation
  initChartBars();

  // Init sidebar toggle for mobile
  initSidebarToggle();
}

// ===== PROFILE INIT =====
function initProfile() {
  initSidebarToggle();

  // Animate progress bars on profile page
  setTimeout(() => {
    document.querySelectorAll('.progress-fill').forEach(bar => {
      const target = bar.getAttribute('data-width');
      bar.style.width = target;
    });
  }, 300);

  initChartBars();
}

// ===== METRICS INIT =====
function initMetrics() {
  // Animate progress bars
  setTimeout(() => {
    document.querySelectorAll('.progress-fill').forEach(bar => {
      const target = bar.getAttribute('data-width');
      bar.style.width = target;
    });
  }, 300);

  // Init chart bars
  initChartBars();

  initSidebarToggle();
}

// ===== CHART BARS ANIMATION =====
function initChartBars() {
  const bars = document.querySelectorAll('.chart-bar');
  bars.forEach((bar, i) => {
    setTimeout(() => {
      const height = bar.getAttribute('data-height');
      bar.style.height = height;
    }, i * 80);
  });
}

// ===== SIDEBAR TOGGLE =====
function initSidebarToggle() {
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

// ===== PASSWORD VISIBILITY TOGGLE =====
function togglePassword(inputId) {
  const input = document.getElementById(inputId);
  if (input) {
    input.type = input.type === 'password' ? 'text' : 'password';
  }
}

// ===== UPLOAD FORM ENHANCEMENTS =====
function initUploadForm() {
  const form = document.getElementById('uploadForm');
  if (!form) return;

  const analyzeBtn = document.getElementById('analyzeBtn');

  form.addEventListener('submit', function () {
    if (analyzeBtn) {
      analyzeBtn.innerHTML = '⏳ Analyzing...';
      analyzeBtn.disabled = true;
      analyzeBtn.style.opacity = '0.7';
    }
  });
}

// ===== INIT ON DOM READY =====
document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initParticles();
  initPawPrints();
  initScrollAnimations();
  initRegisterForm();
  initLoginForm();
  initUploadForm();

  // Page-specific init
  const page = document.body.getAttribute('data-page');
  if (page === 'dashboard') initDashboard();
  if (page === 'profile') initProfile();
  if (page === 'metrics') initMetrics();
  if (page === 'history') initSidebarToggle();

  // Form input focus animations
  document.querySelectorAll('.form-control').forEach(input => {
    input.addEventListener('focus', () => clearFieldError(input));
  });
});

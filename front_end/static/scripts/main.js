/* =====================================================
   MewConnect — Main JS v2.0
   Vanilla JS | No Frameworks | Flask-integrated
   ===================================================== */

// ────────────────────────────────────────────────────
// PAGE TRANSITION OVERLAY
// ────────────────────────────────────────────────────
function initPageTransition() {
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.id = 'page-overlay';
  overlay.style.cssText = `
    position:fixed;inset:0;background:#0a0d14;z-index:99999;
    pointer-events:none;opacity:0;
    transition:opacity 0.35s cubic-bezier(0.4,0,0.2,1);
  `;
  document.body.appendChild(overlay);

  // Fade in on page load
  window.addEventListener('load', () => {
    overlay.style.opacity = '0';
  });

  // Fade out on internal link click
  document.querySelectorAll('a[href]:not([target]):not([href^="#"]):not([href^="mailto"])').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('http') || href.startsWith('//')) return;
      e.preventDefault();
      overlay.style.opacity = '1';
      overlay.style.pointerEvents = 'all';
      setTimeout(() => { window.location.href = href; }, 340);
    });
  });
}

// ────────────────────────────────────────────────────
// NAVBAR
// ────────────────────────────────────────────────────
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  const handleScroll = () => {
    navbar.classList.toggle('scrolled', window.scrollY > 60);
  };
  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll();

  // Mobile hamburger toggle
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.classList.toggle('active', open);
      toggle.setAttribute('aria-expanded', open);
    });
    // Close on outside click
    document.addEventListener('click', e => {
      if (!navbar.contains(e.target)) {
        links.classList.remove('open');
        toggle.classList.remove('active');
      }
    });
  }
}

// ────────────────────────────────────────────────────
// HERO PARTICLES
// ────────────────────────────────────────────────────
function initParticles() {
  const container = document.querySelector('.hero-particles');
  if (!container) return;

  const count = window.innerWidth < 768 ? 18 : 35;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.classList.add('particle');
    const size = Math.random() * 5 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      top:${Math.random() * 100}%;
      animation-delay:${Math.random() * 6}s;
      animation-duration:${Math.random() * 3 + 2.5}s;
    `;
    container.appendChild(p);
  }
}

// ────────────────────────────────────────────────────
// FLOATING PAW PRINTS
// ────────────────────────────────────────────────────
function initPawPrints() {
  const hero = document.querySelector('.hero');
  if (!hero) return;

  const spawn = () => {
    const paw = document.createElement('span');
    paw.textContent = '🐾';
    paw.style.cssText = `
      position:absolute;
      font-size:${Math.random() * 18 + 12}px;
      left:${Math.random() * 95}%;
      top:${Math.random() * 90}%;
      pointer-events:none; z-index:1;
      opacity:0; animation:pawPrint 3s ease forwards;
    `;
    hero.appendChild(paw);
    setTimeout(() => paw.remove(), 3100);
  };
  const interval = setInterval(spawn, 2200);
  // Stop after page hidden
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearInterval(interval);
  });
}

// ────────────────────────────────────────────────────
// SCROLL REVEAL (Intersection Observer)
// ────────────────────────────────────────────────────
function initScrollReveal() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(el => io.observe(el));

  // Legacy support for .animate-on-scroll
  const io2 = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        io2.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
    io2.observe(el);
  });
}

// ────────────────────────────────────────────────────
// COUNTER ANIMATION
// ────────────────────────────────────────────────────
function animateCounter(el, to, duration = 1400) {
  const start = performance.now();
  const from = 0;
  const step = (ts) => {
    const progress = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
    el.textContent = Math.round(from + (to - from) * ease);
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initCounters() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.getAttribute('data-count') || el.textContent, 10);
        if (!isNaN(target)) animateCounter(el, target);
        io.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.stat-card-value, [data-count]').forEach(el => io.observe(el));
}

// ────────────────────────────────────────────────────
// TOAST NOTIFICATIONS
// ────────────────────────────────────────────────────
function showToast(message, type = 'success') {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.innerHTML = `<span style="font-size:1.2rem">${icon}</span><span>${message}</span>`;
  requestAnimationFrame(() => toast.classList.add('show'));
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), 4000);
}
window.showToast = showToast; // expose globally for Jinja flash

// ────────────────────────────────────────────────────
// FORM VALIDATION HELPERS
// ────────────────────────────────────────────────────
function validateEmail(email) { return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email); }
function validatePassword(p) { return p.length >= 6; }

function setFieldError(input, msg) {
  input.style.borderColor = '#ef4444';
  let err = input.parentElement.querySelector('.field-error');
  if (!err) {
    err = document.createElement('span');
    err.className = 'field-error';
    err.style.cssText = 'color:#ef4444;font-size:0.78rem;margin-top:4px;display:block;';
    input.parentElement.appendChild(err);
  }
  err.textContent = msg;
}
function clearFieldError(input) {
  input.style.borderColor = '';
  const err = input.parentElement.querySelector('.field-error');
  if (err) err.remove();
}

// ────────────────────────────────────────────────────
// REGISTER FORM
// ────────────────────────────────────────────────────
function initRegisterForm() {
  const form = document.getElementById('registerForm');
  if (!form) return;

  form.addEventListener('submit', e => {
    let valid = true;
    const name = form.querySelector('#fullName');
    const email = form.querySelector('#email');
    const pass = form.querySelector('#password');
    const conf = form.querySelector('#confirm_password');

    [name, email, pass, conf].filter(Boolean).forEach(clearFieldError);

    if (name && !name.value.trim()) { setFieldError(name, 'Full name is required'); valid = false; }
    if (email && !validateEmail(email.value)) { setFieldError(email, 'Enter a valid email'); valid = false; }
    if (pass && !validatePassword(pass.value)) { setFieldError(pass, 'Min 6 characters required'); valid = false; }
    if (conf && pass && pass.value !== conf.value) { setFieldError(conf, 'Passwords do not match'); valid = false; }

    if (!valid) { e.preventDefault(); return; }

    const btn = form.querySelector('[type=submit]');
    if (btn) { btn.innerHTML = '⏳ Creating Account...'; btn.disabled = true; }
  });

  form.querySelectorAll('.form-control').forEach(input => {
    input.addEventListener('input', () => clearFieldError(input));
  });
}

// ────────────────────────────────────────────────────
// LOGIN FORM
// ────────────────────────────────────────────────────
function initLoginForm() {
  const form = document.getElementById('loginForm');
  if (!form) return;

  form.addEventListener('submit', e => {
    let valid = true;
    const email = form.querySelector('#loginEmail');
    const pass = form.querySelector('#loginPassword');

    [email, pass].forEach(clearFieldError);

    if (!validateEmail(email.value)) { setFieldError(email, 'Enter a valid email'); valid = false; }
    if (!pass.value) { setFieldError(pass, 'Password is required'); valid = false; }

    if (!valid) { e.preventDefault(); return; }

    const btn = form.querySelector('[type=submit]');
    if (btn) { btn.innerHTML = '⏳ Connecting...'; btn.disabled = true; }
  });
}

// ────────────────────────────────────────────────────
// PASSWORD TOGGLE
// ────────────────────────────────────────────────────
function togglePassword(id) {
  const el = document.getElementById(id);
  if (el) el.type = el.type === 'password' ? 'text' : 'password';
}
window.togglePassword = togglePassword;

// ────────────────────────────────────────────────────
// UPLOAD FORM — drag & drop + progress
// ────────────────────────────────────────────────────
function initUploadForm() {
  const form = document.getElementById('uploadForm');
  const dropZone = document.getElementById('dropZone');
  const fileIn = document.getElementById('file');
  const fileLabel = document.getElementById('fileLabel');
  const btn = document.getElementById('analyzeBtn');

  if (!form) return;

  // Drag & Drop
  if (dropZone && fileIn) {
    ['dragenter', 'dragover'].forEach(ev => {
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    });
    dropZone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files.length) {
        fileIn.files = files;
        if (fileLabel) fileLabel.textContent = '✅ ' + files[0].name;
      }
    });
    dropZone.addEventListener('click', () => fileIn.click());
  }

  if (fileIn && fileLabel) {
    fileIn.addEventListener('change', () => {
      if (fileIn.files.length) fileLabel.textContent = '✅ ' + fileIn.files[0].name;
    });
  }

  form.addEventListener('submit', () => {
    if (btn) { btn.innerHTML = '⏳ Analyzing...'; btn.disabled = true; btn.style.opacity = '0.75'; }
  });
}

// ────────────────────────────────────────────────────
// CHART BARS ANIMATION
// ────────────────────────────────────────────────────
function initChartBars() {
  document.querySelectorAll('.chart-bar').forEach((bar, i) => {
    const h = bar.getAttribute('data-height') || '50%';
    bar.style.height = '0';
    setTimeout(() => { bar.style.height = h; }, i * 90);
  });
}

// ────────────────────────────────────────────────────
// PROGRESS BAR ANIMATION
// ────────────────────────────────────────────────────
function initProgressBars() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const w = entry.target.getAttribute('data-width') || '0%';
        entry.target.style.width = w;
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll('.progress-fill').forEach(el => io.observe(el));
}

// ────────────────────────────────────────────────────
// SIDEBAR TOGGLE (mobile)
// ────────────────────────────────────────────────────
function initSidebarToggle() {
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (!toggle || !sidebar) return;

  toggle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    if (overlay) overlay.style.display = open ? 'block' : 'none';
  });
  if (overlay) {
    overlay.addEventListener('click', () => {
      sidebar.classList.remove('open');
      overlay.style.display = 'none';
    });
  }
}

// ────────────────────────────────────────────────────
// LAZY VIDEO LOADING
// ────────────────────────────────────────────────────
function initLazyVideos() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const video = entry.target;
        video.querySelectorAll('source').forEach(source => {
          if (source.dataset.src) {
            source.src = source.dataset.src;
            source.removeAttribute('data-src');
          }
        });
        video.load();
        io.unobserve(video);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('video[data-lazy]').forEach(v => io.observe(v));
}

// ────────────────────────────────────────────────────
// VIDEO HOVER PLAY/PAUSE
// ────────────────────────────────────────────────────
function initVideoHover() {
  document.querySelectorAll('.video-card-wrap video, .gallery-card video').forEach(video => {
    const parent = video.closest('[data-video-hover]');
    if (!parent) return;
    parent.addEventListener('mouseenter', () => video.play());
    parent.addEventListener('mouseleave', () => { video.pause(); });
  });
}

// ────────────────────────────────────────────────────
// SMOOTH SCROLL for anchor links
// ────────────────────────────────────────────────────
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ────────────────────────────────────────────────────
// MOBILE RESPONSIVE — sidebar show/hide on resize
// ────────────────────────────────────────────────────
function initResponsive() {
  const toggle = document.getElementById('sidebarToggle');
  if (!toggle) return;
  const show = () => { toggle.style.display = window.innerWidth <= 768 ? 'flex' : 'none'; };
  window.addEventListener('resize', show);
  show();
}

// ────────────────────────────────────────────────────
// HERO NUMBER STATS COUNTER
// ────────────────────────────────────────────────────
function initHeroStats() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const to = parseInt(el.getAttribute('data-count'), 10);
      if (!isNaN(to)) animateCounter(el, to);
      io.unobserve(el);
    });
  }, { threshold: 0.6 });
  document.querySelectorAll('.hero-stat-num[data-count]').forEach(el => io.observe(el));
}

// ────────────────────────────────────────────────────
// ACTIVE NAV LINK based on scroll
// ────────────────────────────────────────────────────
function initActiveNav() {
  const sections = document.querySelectorAll('section[id]');
  if (!sections.length) return;
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.nav-links a').forEach(a => {
          a.classList.toggle('active-link', a.getAttribute('href') === '#' + entry.target.id);
        });
      }
    });
  }, { threshold: 0.5 });
  sections.forEach(s => io.observe(s));
}

// ────────────────────────────────────────────────────
// PAGE-SPECIFIC INIT
// ────────────────────────────────────────────────────
function initDashboard() {
  initChartBars();
  initSidebarToggle();
  initResponsive();
  initCounters();
}

function initProfile() {
  initSidebarToggle();
  initResponsive();
  initProgressBars();
  initChartBars();
}

function initMetrics() {
  initSidebarToggle();
  initResponsive();
  initProgressBars();
  initChartBars();
}

function initHistory() {
  initSidebarToggle();
  initResponsive();
}

// ────────────────────────────────────────────────────
// MAIN INIT
// ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initPageTransition();
  initNavbar();
  initParticles();
  initPawPrints();
  initScrollReveal();
  initSmoothScroll();
  initRegisterForm();
  initLoginForm();
  initUploadForm();
  initLazyVideos();
  initHeroStats();
  initActiveNav();
  initProgressBars();

  const page = document.body.getAttribute('data-page');
  if (page === 'dashboard') initDashboard();
  if (page === 'profile') initProfile();
  if (page === 'metrics') initMetrics();
  if (page === 'history') initHistory();

  // Input focus helpers
  document.querySelectorAll('.form-control').forEach(input => {
    input.addEventListener('focus', () => clearFieldError(input));
  });
});

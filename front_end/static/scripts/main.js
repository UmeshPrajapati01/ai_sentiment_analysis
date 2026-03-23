/* =====================================================
   MeowMood — Main JS v3.0
   ===================================================== */

// PAGE TRANSITION
function initPageTransition() {
  const overlay = document.createElement('div');
  overlay.id = 'page-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:#060810;z-index:99999;pointer-events:none;opacity:0;transition:opacity 0.35s cubic-bezier(0.4,0,0.2,1);';
  document.body.appendChild(overlay);
  window.addEventListener('load', () => { overlay.style.opacity = '0'; overlay.style.pointerEvents = 'none'; });
  // Fade in on pageshow (handles back/forward cache)
  window.addEventListener('pageshow', () => { overlay.style.opacity = '0'; overlay.style.pointerEvents = 'none'; });
  document.querySelectorAll('a[href]:not([target]):not([href^="#"]):not([href^="mailto"]):not([onclick])').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('http') || href.startsWith('//')) return;
      // Skip same-page hash navigation
      const url = new URL(href, window.location.href);
      if (url.pathname === window.location.pathname && url.hash) return;
      e.preventDefault();
      overlay.style.opacity = '1'; overlay.style.pointerEvents = 'all';
      setTimeout(() => { window.location.href = href; }, 340);
    });
  });
}

// NAVBAR
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;
  const handleScroll = () => navbar.classList.toggle('scrolled', window.scrollY > 60);
  window.addEventListener('scroll', handleScroll, { passive: true }); handleScroll();
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.classList.toggle('active', open);
      toggle.setAttribute('aria-expanded', open);
    });
    document.addEventListener('click', e => {
      if (!navbar.contains(e.target)) { links.classList.remove('open'); toggle.classList.remove('active'); }
    });
  }
}

// PARTICLES
function initParticles() {
  const container = document.querySelector('.hero-particles');
  if (!container) return;
  const count = window.innerWidth < 768 ? 18 : 40;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div'); p.classList.add('particle');
    const size = Math.random() * 5 + 2;
    p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;top:${Math.random()*100}%;animation-delay:${Math.random()*6}s;animation-duration:${Math.random()*3+2.5}s;`;
    container.appendChild(p);
  }
}

// SCROLL REVEAL
function initScrollReveal() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('revealed'); io.unobserve(entry.target); } });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(el => io.observe(el));
}

// COUNTER ANIMATION
function animateCounter(el, to, duration = 1400) {
  const start = performance.now();
  const step = (ts) => {
    const progress = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(to * ease);
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

// TOAST
function showToast(message, type = 'success') {
  let toast = document.querySelector('.toast');
  if (!toast) { toast = document.createElement('div'); toast.className = 'toast'; document.body.appendChild(toast); }
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.innerHTML = `<span style="font-size:1.2rem">${icon}</span><span>${message}</span>`;
  requestAnimationFrame(() => toast.classList.add('show'));
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), 4000);
}
window.showToast = showToast;

// FORM VALIDATION
function validateEmail(email) { return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email); }
function setFieldError(input, msg) {
  input.style.borderColor = '#ef4444';
  let err = input.parentElement.querySelector('.field-error');
  if (!err) { err = document.createElement('span'); err.className = 'field-error'; err.style.cssText = 'color:#ef4444;font-size:0.78rem;margin-top:4px;display:block;'; input.parentElement.appendChild(err); }
  err.textContent = msg;
}
function clearFieldError(input) { input.style.borderColor = ''; const err = input.parentElement.querySelector('.field-error'); if (err) err.remove(); }

function initRegisterForm() {
  const form = document.getElementById('registerForm');
  if (!form) return;
  form.addEventListener('submit', e => {
    let valid = true;
    const name = form.querySelector('#fullName'), email = form.querySelector('#email');
    const pass = form.querySelector('#password'), conf = form.querySelector('#confirm_password');
    [name, email, pass, conf].filter(Boolean).forEach(clearFieldError);
    if (name && !name.value.trim()) { setFieldError(name, 'Full name is required'); valid = false; }
    if (email && !validateEmail(email.value)) { setFieldError(email, 'Enter a valid email'); valid = false; }
    if (pass && pass.value.length < 6) { setFieldError(pass, 'Min 6 characters required'); valid = false; }
    if (conf && pass && pass.value !== conf.value) { setFieldError(conf, 'Passwords do not match'); valid = false; }
    if (!valid) { e.preventDefault(); return; }
    const btn = form.querySelector('[type=submit]');
    if (btn) { btn.innerHTML = '⏳ Creating Account...'; btn.disabled = true; }
  });
  form.querySelectorAll('.form-control').forEach(input => input.addEventListener('input', () => clearFieldError(input)));
}

function initLoginForm() {
  const form = document.getElementById('loginForm');
  if (!form) return;
  form.addEventListener('submit', e => {
    let valid = true;
    const email = form.querySelector('#loginEmail'), pass = form.querySelector('#loginPassword');
    [email, pass].forEach(clearFieldError);
    if (!validateEmail(email.value)) { setFieldError(email, 'Enter a valid email'); valid = false; }
    if (!pass.value) { setFieldError(pass, 'Password is required'); valid = false; }
    if (!valid) { e.preventDefault(); return; }
    const btn = form.querySelector('[type=submit]');
    if (btn) { btn.innerHTML = '⏳ Connecting...'; btn.disabled = true; }
  });
}

function togglePassword(id) { const el = document.getElementById(id); if (el) el.type = el.type === 'password' ? 'text' : 'password'; }
window.togglePassword = togglePassword;

// CHART BARS
function initChartBars() {
  document.querySelectorAll('.chart-bar').forEach((bar, i) => {
    const h = bar.getAttribute('data-height') || '50%';
    bar.style.height = '0';
    setTimeout(() => { bar.style.height = h; }, i * 90);
  });
}

// PROGRESS BARS
function initProgressBars() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.style.width = entry.target.getAttribute('data-width') || '0%'; io.unobserve(entry.target); }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll('.progress-fill').forEach(el => io.observe(el));
}

// SIDEBAR TOGGLE
function initSidebarToggle() {
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (!toggle || !sidebar) return;
  toggle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    if (overlay) overlay.style.display = open ? 'block' : 'none';
  });
  if (overlay) overlay.addEventListener('click', () => { sidebar.classList.remove('open'); overlay.style.display = 'none'; });
}

function initResponsive() {
  const toggle = document.getElementById('sidebarToggle');
  if (!toggle) return;
  const show = () => { toggle.style.display = window.innerWidth <= 768 ? 'flex' : 'none'; };
  window.addEventListener('resize', show); show();
}

// SMOOTH SCROLL
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    });
  });
}

// HERO STATS
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

// MAIN INIT
document.addEventListener('DOMContentLoaded', () => {
  initPageTransition();
  initNavbar();
  initParticles();
  initScrollReveal();
  initSmoothScroll();
  initRegisterForm();
  initLoginForm();
  initHeroStats();
  initProgressBars();

  const page = document.body.getAttribute('data-page');
  if (page === 'dashboard') { initChartBars(); initSidebarToggle(); initResponsive(); initCounters(); }
  if (page === 'profile')   { initSidebarToggle(); initResponsive(); initProgressBars(); }
  if (page === 'metrics')   { initSidebarToggle(); initResponsive(); initProgressBars(); initChartBars(); }
  if (page === 'history')   { initSidebarToggle(); initResponsive(); }

  document.querySelectorAll('.form-control').forEach(input => input.addEventListener('focus', () => clearFieldError(input)));
});

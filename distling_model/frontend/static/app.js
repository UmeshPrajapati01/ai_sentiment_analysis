/**
 * ═══════════════════════════════════════════════════════
 *  MewSense — Cat Emotion Analyzer
 *  Application Logic v5 — Vanilla ES6+
 *  Bulletproof file upload + dynamic UI + full API integration
 * ═══════════════════════════════════════════════════════
 */

'use strict';

const CONFIG = Object.freeze({
    API_BASE: window.location.origin,
    HEALTH_INTERVAL: 25000,
    WEBCAM_INTERVAL: 3000,
    PARTICLE_COUNT: 55,
    MAX_HISTORY: 10,
    MAX_FILE_SIZE: 10 * 1024 * 1024,
    FETCH_TIMEOUT: 30000,
    IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/webp'],
    AUDIO_TYPES: ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-wav'],
    IMAGE_EXTS: ['jpg', 'jpeg', 'png', 'webp'],
    AUDIO_EXTS: ['wav', 'mp3', 'flac'],
});

const state = {
    mode: 'image',
    file: null,
    analyzing: false,
    webcamStream: null,
    webcamTimer: null,
    history: [],
};


// ─────────────────────────────────────────────────────────────
// DOM CACHE
// ─────────────────────────────────────────────────────────────
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
let D = {};

function cacheDom() {
    D = {
        tabs: $$('.mode-tab'),
        tabBg: $('#tabBg'),

        viewImage: $('#viewImage'),
        viewAudio: $('#viewAudio'),
        viewWebcam: $('#viewWebcam'),

        uploadImage: $('#uploadZoneImage'),
        uploadAudio: $('#uploadZoneAudio'),
        fileImage: $('#fileImage'),
        fileAudio: $('#fileAudio'),

        previewImage: $('#previewImage'),
        previewImageEl: $('#previewImageEl'),
        closePreviewImage: $('#closePreviewImage'),
        previewAudio: $('#previewAudio'),
        audioName: $('#audioName'),
        audioSize: $('#audioSize'),
        closePreviewAudio: $('#closePreviewAudio'),

        webcamVideo: $('#webcamVideo'),
        webcamCanvas: $('#webcamCanvas'),
        camOverlay: $('#camOverlay'),
        camEmoji: $('#camEmoji'),
        camLabel: $('#camLabel'),
        camConf: $('#camConf'),
        btnCamStart: $('#btnCamStart'),
        btnCamCapture: $('#btnCamCapture'),
        btnCamStop: $('#btnCamStop'),

        btnAnalyze: $('#btnAnalyze'),

        resultsEmpty: $('#resultsEmpty'),
        resultsData: $('#resultsData'),
        ringProgress: $('#ringProgress'),
        ringEmoji: $('#ringEmoji'),
        emotionName: $('#emotionName'),
        emotionDesc: $('#emotionDesc'),
        confNum: $('#confNum'),
        probsList: $('#probsList'),
        tipsBox: $('#tipsBox'),
        tipsContent: $('#tipsContent'),
        metaChips: $('#metaChips'),
        demoBanner: $('#demoBanner'),

        historySection: $('#historySection'),
        historyList: $('#historyList'),

        statusIndicator: $('#statusIndicator'),
        statusText: $('#statusText'),

        meshCanvas: $('#meshCanvas'),
    };
}


// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    cacheDom();
    initMeshBg();
    initTabs();
    initUploads();
    initWebcam();
    initAnalyze();
    initScrollReveal();
    initTiltCards();
    initNavScroll();
    checkHealth();
    setInterval(checkHealth, CONFIG.HEALTH_INTERVAL);

    // Position tab bg after fonts load
    requestAnimationFrame(() => requestAnimationFrame(posTabBg));
});


// ─────────────────────────────────────────────────────────────
// SCROLL REVEAL (IntersectionObserver)
// ─────────────────────────────────────────────────────────────
function initScrollReveal() {
    const cards = document.querySelectorAll('.reveal-card');
    if (!cards.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = parseInt(entry.target.dataset.delay) || 0;
                setTimeout(() => entry.target.classList.add('in-view'), delay);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

    cards.forEach(c => observer.observe(c));
}


// ─────────────────────────────────────────────────────────────
// PARALLAX TILT ON GALLERY CARDS
// ─────────────────────────────────────────────────────────────
function initTiltCards() {
    document.querySelectorAll('.masonry-item').forEach(card => {
        card.addEventListener('mousemove', e => {
            const r = card.getBoundingClientRect();
            const x = (e.clientX - r.left) / r.width - 0.5;
            const y = (e.clientY - r.top) / r.height - 0.5;
            card.style.transform = `perspective(600px) rotateY(${x * 8}deg) rotateX(${-y * 8}deg) scale(1.03)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}


// ─────────────────────────────────────────────────────────────
// NAV SCROLL EFFECT
// ─────────────────────────────────────────────────────────────
function initNavScroll() {
    const nav = document.getElementById('mainNav');
    if (!nav) return;
    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                nav.classList.toggle('scrolled', window.scrollY > 50);
                ticking = false;
            });
            ticking = true;
        }
    });
}


// ─────────────────────────────────────────────────────────────
// MESH BACKGROUND (animated particles + connections)
// ─────────────────────────────────────────────────────────────
function initMeshBg() {
    const canvas = D.meshCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let pts = [], w, h;

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    function create() {
        return {
            x: Math.random() * w,
            y: Math.random() * h,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            r: Math.random() * 1.6 + 0.5,
            a: Math.random() * 0.35 + 0.08,
            hue: Math.random() > 0.6 ? 45 : 270, // gold or purple
        };
    }

    function init() {
        resize();
        pts = Array.from({ length: CONFIG.PARTICLE_COUNT }, create);
    }

    function loop() {
        ctx.clearRect(0, 0, w, h);
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 75%, 65%, ${p.a})`;
            ctx.fill();

            for (let j = i + 1; j < pts.length; j++) {
                const q = pts[j];
                const dx = p.x - q.x, dy = p.y - q.y;
                const d = Math.sqrt(dx * dx + dy * dy);
                if (d < 130) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(q.x, q.y);
                    const alpha = 0.05 * (1 - d / 130);
                    ctx.strokeStyle = `hsla(${p.hue}, 50%, 55%, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(loop);
    }

    init();
    loop();
    window.addEventListener('resize', resize);
}


// ─────────────────────────────────────────────────────────────
// TABS
// ─────────────────────────────────────────────────────────────
function initTabs() {
    D.tabs.forEach(t => t.addEventListener('click', () => switchMode(t.dataset.mode)));
}

function switchMode(mode) {
    state.mode = mode;
    state.file = null;
    D.btnAnalyze.disabled = true;

    D.tabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
    posTabBg();

    D.viewImage.classList.toggle('active', mode === 'image');
    D.viewAudio.classList.toggle('active', mode === 'audio');
    D.viewWebcam.classList.toggle('active', mode === 'webcam');

    D.previewImage.classList.remove('visible');
    D.previewAudio.classList.remove('visible');

    if (mode !== 'webcam') stopCam();
    D.btnAnalyze.style.display = mode === 'webcam' ? 'none' : '';
}

function posTabBg() {
    const bar = $('#modeTabs');
    const active = bar ? bar.querySelector('.mode-tab.active') : null;
    if (!active || !D.tabBg) return;
    const bR = bar.getBoundingClientRect();
    const tR = active.getBoundingClientRect();
    D.tabBg.style.width = tR.width + 'px';
    D.tabBg.style.left = (tR.left - bR.left) + 'px';
}

window.addEventListener('resize', posTabBg);


// ─────────────────────────────────────────────────────────────
// FILE UPLOADS — BULLETPROOF
// The <input type="file"> sits as a full-coverage overlay
// on the upload zone, so ANY click triggers native dialog.
// We only need JS for: drag events + change event + preview.
// ─────────────────────────────────────────────────────────────
function initUploads() {
    // IMAGE
    setupUpload(D.uploadImage, D.fileImage, 'image');
    D.closePreviewImage.addEventListener('click', () => {
        state.file = null;
        D.previewImage.classList.remove('visible');
        D.btnAnalyze.disabled = true;
        D.fileImage.value = '';
    });

    // AUDIO
    setupUpload(D.uploadAudio, D.fileAudio, 'audio');
    D.closePreviewAudio.addEventListener('click', () => {
        state.file = null;
        D.previewAudio.classList.remove('visible');
        D.btnAnalyze.disabled = true;
        D.fileAudio.value = '';
    });
}

function setupUpload(zone, input, type) {
    if (!zone || !input) return;

    // ── Native file input change (works for BOTH click and drag) ──
    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0], type);
    });

    // ── Drag visual feedback ──
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', e => { e.preventDefault(); zone.classList.remove('dragover'); });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], type);
    });
}

function handleFile(file, type) {
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = type === 'image' ? CONFIG.IMAGE_EXTS : CONFIG.AUDIO_EXTS;
    const allowedMimes = type === 'image' ? CONFIG.IMAGE_TYPES : CONFIG.AUDIO_TYPES;

    if (!allowedMimes.includes(file.type) && !allowedExts.includes(ext)) {
        toast(`Unsupported format. Use ${allowedExts.join(', ').toUpperCase()}`);
        return;
    }

    if (file.size > CONFIG.MAX_FILE_SIZE) {
        toast('File too large. Max 10 MB.');
        return;
    }

    state.file = file;

    if (type === 'image') {
        const reader = new FileReader();
        reader.onload = e => {
            D.previewImageEl.src = e.target.result;
            D.previewImage.classList.add('visible');
        };
        reader.readAsDataURL(file);
    } else {
        D.audioName.textContent = file.name;
        D.audioSize.textContent = fmtBytes(file.size);
        D.previewAudio.classList.add('visible');
    }

    D.btnAnalyze.disabled = false;
    console.log(`[MewSense] Selected: ${file.name} (${fmtBytes(file.size)})`);
}


// ─────────────────────────────────────────────────────────────
// WEBCAM
// ─────────────────────────────────────────────────────────────
function initWebcam() {
    D.btnCamStart?.addEventListener('click', startCam);
    D.btnCamStop?.addEventListener('click', stopCam);
    D.btnCamCapture?.addEventListener('click', captureCam);
}

async function startCam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });
        state.webcamStream = stream;
        D.webcamVideo.srcObject = stream;
        D.btnCamStart.disabled = true;
        D.btnCamCapture.disabled = false;
        D.btnCamStop.disabled = false;
        state.webcamTimer = setInterval(autoCam, CONFIG.WEBCAM_INTERVAL);
    } catch (err) {
        toast('Webcam access denied. Check permissions.');
        console.error(err);
    }
}

function stopCam() {
    state.webcamStream?.getTracks().forEach(t => t.stop());
    state.webcamStream = null;
    if (state.webcamTimer) { clearInterval(state.webcamTimer); state.webcamTimer = null; }
    D.webcamVideo.srcObject = null;
    D.btnCamStart.disabled = false;
    D.btnCamCapture.disabled = true;
    D.btnCamStop.disabled = true;
    D.camOverlay?.classList.remove('visible');
}

function grabFrame() {
    const v = D.webcamVideo, c = D.webcamCanvas;
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
}

async function captureCam() {
    if (state.analyzing) return;
    const blob = await grabFrame();
    await predict(new File([blob], 'webcam.jpg', { type: 'image/jpeg' }), 'image');
}

async function autoCam() {
    if (state.analyzing || !state.webcamStream) return;
    const blob = await grabFrame();
    await predict(new File([blob], 'webcam.jpg', { type: 'image/jpeg' }), 'image', true);
}


// ─────────────────────────────────────────────────────────────
// ANALYZE BUTTON
// ─────────────────────────────────────────────────────────────
function initAnalyze() {
    D.btnAnalyze.addEventListener('click', () => {
        if (!state.file || state.analyzing) return;
        predict(state.file, state.mode === 'audio' ? 'audio' : 'image');
    });
}


// ─────────────────────────────────────────────────────────────
// API PREDICTION
// ─────────────────────────────────────────────────────────────
async function predict(file, modality, auto = false) {
    state.analyzing = true;
    if (!auto) {
        D.btnAnalyze.classList.add('loading');
        D.btnAnalyze.disabled = true;
    }

    const fd = new FormData();
    fd.append('file', file);
    const url = CONFIG.API_BASE + (modality === 'audio' ? '/predict/audio' : '/predict/image');

    try {
        const ctrl = new AbortController();
        const timeout = setTimeout(() => ctrl.abort(), CONFIG.FETCH_TIMEOUT);

        const resp = await fetch(url, { method: 'POST', body: fd, signal: ctrl.signal });
        clearTimeout(timeout);

        if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);

        const result = await resp.json();
        showResults(result);
        if (state.mode === 'webcam') showCamOverlay(result);
        addHistory(result, file.name);
    } catch (err) {
        toast(err.name === 'AbortError' ? 'Request timed out.' : 'Analysis failed: ' + err.message);
        console.error(err);
    } finally {
        state.analyzing = false;
        if (!auto) {
            D.btnAnalyze.classList.remove('loading');
            D.btnAnalyze.disabled = !state.file;
        }
    }
}


// ─────────────────────────────────────────────────────────────
// DISPLAY RESULTS
// ─────────────────────────────────────────────────────────────
function showResults(r) {
    D.resultsEmpty.style.display = 'none';
    D.resultsData.classList.add('visible');

    setAccent(r.prediction);

    // Emoji + pop animation
    D.ringEmoji.textContent = r.emoji || '🐱';
    D.ringEmoji.classList.remove('pop');
    void D.ringEmoji.offsetHeight;
    D.ringEmoji.classList.add('pop');

    D.emotionName.textContent = r.prediction || '—';
    D.emotionDesc.textContent = r.description || '';

    // Confidence ring
    const conf = r.confidence || 0;
    const circ = 2 * Math.PI * 62; // r=62
    D.ringProgress.style.strokeDashoffset = circ * (1 - conf);

    animNum(D.confNum, Math.round(conf * 100), 900);

    // Probability bars
    renderProbs(r.all_probabilities);

    // Tips
    if (r.tips) {
        D.tipsBox.classList.add('visible');
        D.tipsContent.textContent = r.tips;
    } else {
        D.tipsBox.classList.remove('visible');
    }

    // Meta
    D.metaChips.innerHTML = `
        <span class="meta-chip">Model <b>v${r.model_version || '?'}</b></span>
        <span class="meta-chip">Acc <b>${Math.round((r.model_accuracy || 0) * 100)}%</b></span>
        <span class="meta-chip">Latency <b>${r.latency_ms || '?'}ms</b></span>
        <span class="meta-chip"><b>${r.modality || state.mode}</b></span>
    `;

    D.demoBanner.classList.toggle('visible', !!r.demo_mode);
}

function renderProbs(probs) {
    if (!probs) { D.probsList.innerHTML = ''; return; }
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const topName = sorted[0]?.[0];

    D.probsList.innerHTML = '';
    sorted.forEach(([name, prob]) => {
        const pct = Math.round(prob * 100);
        const isTop = name === topName;
        const row = document.createElement('div');
        row.className = 'prob-row' + (isTop ? ' top' : '');
        row.innerHTML = `
            <span class="prob-label">${name}</span>
            <div class="prob-track">
                <div class="prob-bar ${isTop ? 'main' : 'dim'}" style="width:0%"></div>
            </div>
            <span class="prob-pct">${pct}%</span>
        `;
        D.probsList.appendChild(row);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            row.querySelector('.prob-bar').style.width = pct + '%';
        }));
    });
}

function animNum(el, target, dur) {
    const start = parseInt(el.textContent) || 0;
    const range = target - start;
    const t0 = performance.now();
    (function tick(now) {
        const p = Math.min((now - t0) / dur, 1);
        el.textContent = Math.round(start + range * (1 - Math.pow(1 - p, 3)));
        if (p < 1) requestAnimationFrame(tick);
    })(t0);
}


// ─────────────────────────────────────────────────────────────
// EMOTION ACCENT
// ─────────────────────────────────────────────────────────────
function setAccent(pred) {
    [...document.body.classList].filter(c => c.startsWith('emo-')).forEach(c => document.body.classList.remove(c));
    document.body.classList.add('emo-' + (pred || '').toLowerCase().replace(/\s+/g, ''));
}


// ─────────────────────────────────────────────────────────────
// WEBCAM OVERLAY
// ─────────────────────────────────────────────────────────────
function showCamOverlay(r) {
    D.camEmoji.textContent = r.emoji || '🐱';
    D.camLabel.textContent = r.prediction || '—';
    D.camConf.textContent = Math.round((r.confidence || 0) * 100) + '%';
    D.camOverlay.classList.add('visible');
}


// ─────────────────────────────────────────────────────────────
// HISTORY
// ─────────────────────────────────────────────────────────────
function addHistory(r, fname) {
    state.history.unshift({
        prediction: r.prediction,
        confidence: r.confidence,
        emoji: r.emoji || '🐱',
        modality: r.modality || state.mode,
        file: fname,
        time: new Date().toLocaleTimeString(),
    });
    if (state.history.length > CONFIG.MAX_HISTORY) state.history.pop();
    renderHistory();
}

function renderHistory() {
    if (!state.history.length) { D.historySection.classList.remove('visible'); return; }
    D.historySection.classList.add('visible');
    D.historyList.innerHTML = '';
    state.history.forEach(h => {
        const card = document.createElement('div');
        card.className = 'hist-card';
        card.innerHTML = `
            <span class="hist-emoji">${h.emoji}</span>
            <div class="hist-meta">
                <div class="hist-pred">${h.prediction}</div>
                <div class="hist-detail">${h.file} · ${h.time} · ${h.modality}</div>
            </div>
            <span class="hist-conf">${Math.round(h.confidence * 100)}%</span>
        `;
        D.historyList.appendChild(card);
    });
}


// ─────────────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await (await fetch(CONFIG.API_BASE + '/health')).json();
        D.statusIndicator.className = 'status-indicator on';
        const parts = [];
        if (r.audio_model_loaded) parts.push('Audio v' + r.audio_model_version);
        if (r.image_model_loaded) parts.push('Image v' + r.image_model_version);
        D.statusText.textContent = parts.length ? parts.join(' · ') : 'Online';
    } catch {
        D.statusIndicator.className = 'status-indicator off';
        D.statusText.textContent = 'Offline';
    }
}


// ─────────────────────────────────────────────────────────────
// TOAST
// ─────────────────────────────────────────────────────────────
function toast(msg) {
    const el = document.createElement('div');
    el.style.cssText = `
        position:fixed; bottom:28px; left:50%; transform:translateX(-50%) translateY(16px);
        padding:14px 28px; background:rgba(239,68,68,0.92); color:white; font-size:14px;
        font-weight:700; font-family:var(--ff-body); border-radius:14px;
        backdrop-filter:blur(12px); box-shadow:0 8px 32px rgba(239,68,68,0.3);
        z-index:9999; opacity:0; transition:all 0.4s cubic-bezier(0.16,1,0.3,1);
        max-width:90vw; text-align:center;
    `;
    el.textContent = msg;
    document.body.appendChild(el);
    requestAnimationFrame(() => { el.style.opacity = '1'; el.style.transform = 'translateX(-50%) translateY(0)'; });
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(-50%) translateY(16px)';
        setTimeout(() => el.remove(), 400);
    }, 4000);
}


// ─────────────────────────────────────────────────────────────
// UTILS
// ─────────────────────────────────────────────────────────────
function fmtBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
}

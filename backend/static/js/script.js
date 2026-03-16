/* ═══════════════════════════════════════════
   PARTICLE / MOUSE TRAIL
═══════════════════════════════════════════ */
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');
let W = canvas.width = window.innerWidth;
let H = canvas.height = window.innerHeight;
window.addEventListener('resize', () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; });

const particles = [];
window.addEventListener('mousemove', e => {
    for (let i = 0; i < 3; i++) spawnParticle(e.clientX, e.clientY);
});

function spawnParticle(x, y) {
    const colors = ['#a78bfa', '#34d399', '#60a5fa', '#f472b6', '#fbbf24'];
    particles.push({
        x, y,
        vx: (Math.random() - 0.5) * 2.5,
        vy: (Math.random() - 0.5) * 2.5 - 1,
        life: 1,
        decay: 0.018 + Math.random() * 0.02,
        size: 2 + Math.random() * 4,
        color: colors[Math.floor(Math.random() * colors.length)]
    });
}

for (let i = 0; i < 40; i++) {
    particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.3,
        vy: -0.2 - Math.random() * 0.3,
        life: Math.random(),
        decay: 0.001 + Math.random() * 0.002,
        size: 1 + Math.random() * 2,
        color: '#a78bfa',
        ambient: true
    });
}

function animateParticles() {
    ctx.clearRect(0, 0, W, H);
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy; p.life -= p.decay;
        if (p.ambient && p.life <= 0) { p.x = Math.random() * W; p.y = H + 10; p.life = 0.6 + Math.random() * 0.4; }
        else if (!p.ambient && p.life <= 0) { particles.splice(i, 1); continue; }
        ctx.save();
        ctx.globalAlpha = p.life * (p.ambient ? 0.4 : 0.85);
        ctx.fillStyle = p.color;
        ctx.shadowBlur = p.ambient ? 4 : 10;
        ctx.shadowColor = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
    requestAnimationFrame(animateParticles);
}
animateParticles();

/* ═══════════════════════════════════════════
   TOAST NOTIFICATIONS
═══════════════════════════════════════════ */
function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 400); }, 3500);
}

/* ═══════════════════════════════════════════
   AUTH LOGIC
═══════════════════════════════════════════ */
function showForm(id) {
    document.querySelectorAll('.auth-form').forEach(f => f.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

function togglePw(inputId, btn) {
    const input = document.getElementById(inputId);
    if (input.type === 'password') { input.type = 'text'; btn.textContent = '🙈'; }
    else { input.type = 'password'; btn.textContent = '👁'; }
}

function checkStrength(pw) {
    const fill = document.getElementById('pw-strength-fill');
    const label = document.getElementById('pw-strength-label');
    const hints = { len: pw.length >= 8, upper: /[A-Z]/.test(pw), lower: /[a-z]/.test(pw), num: /[0-9]/.test(pw), special: /[^A-Za-z0-9]/.test(pw) };
    Object.keys(hints).forEach(k => { const el = document.getElementById(`hint-${k}`); if (el) el.classList.toggle('valid', hints[k]); });
    const score = Object.values(hints).filter(Boolean).length;
    const levels = [
        { w: '0%',   bg: 'transparent', txt: 'Enter a password' },
        { w: '20%',  bg: '#f87171',     txt: 'Very weak' },
        { w: '40%',  bg: '#fbbf24',     txt: 'Weak' },
        { w: '60%',  bg: '#facc15',     txt: 'Fair' },
        { w: '80%',  bg: '#34d399',     txt: 'Strong' },
        { w: '100%', bg: 'linear-gradient(90deg,#34d399,#60a5fa)', txt: 'Very strong' }
    ];
    const lvl = levels[score];
    fill.style.width = lvl.w; fill.style.background = lvl.bg; label.textContent = lvl.txt;
    label.style.color = score >= 4 ? '#34d399' : score >= 2 ? '#fbbf24' : '#f87171';
}

function handleLogin() {
    const email = document.getElementById('login-email').value.trim();
    const pw    = document.getElementById('login-password').value;
    if (!email || !pw) { shake(); return; }
    const stored = JSON.parse(localStorage.getItem('mm_user') || 'null');
    if (stored && stored.email === email && stored.pw === pw) {
        loginSuccess(stored.name || email.split('@')[0]);
    } else if (!stored) {
        loginSuccess(email.split('@')[0]);
    } else {
        shake(); showAuthError('Invalid email or password.');
    }
}

function handleSignup() {
    const name  = document.getElementById('signup-name').value.trim();
    const email = document.getElementById('signup-email').value.trim();
    const pw    = document.getElementById('signup-password').value;
    if (!name || !email || !pw) { shake(); return; }
    if (pw.length < 8) { shake(); showAuthError('Password must be at least 8 characters.'); return; }
    localStorage.setItem('mm_user', JSON.stringify({ name, email, pw }));
    loginSuccess(name);
}

function handleForgot() {
    const email = document.getElementById('forgot-email').value.trim();
    if (!email) { shake(); return; }
    document.getElementById('forgot-success').classList.remove('hidden');
}

function loginSuccess(name) {
    const overlay = document.getElementById('auth-overlay');
    overlay.classList.add('fade-out');
    setTimeout(() => { overlay.style.display = 'none'; }, 400);
    document.getElementById('main-app').style.display = 'block';
    document.getElementById('user-greeting').textContent = `Hi, ${name} 👋`;
    renderHistory();
}

function handleLogout() {
    document.getElementById('main-app').style.display = 'none';
    const overlay = document.getElementById('auth-overlay');
    overlay.style.display = 'flex';
    overlay.classList.remove('fade-out');
    showForm('form-login');
}

function shake() {
    const card = document.getElementById('auth-card');
    card.style.animation = 'none'; card.offsetHeight;
    card.style.animation = 'shake 0.4s ease';
}

function showAuthError(msg) {
    let err = document.getElementById('auth-error');
    if (!err) {
        err = document.createElement('div');
        err.id = 'auth-error';
        err.style.cssText = 'color:#f87171;font-size:0.85rem;text-align:center;animation:fadeIn 0.3s ease;';
        document.querySelector('.auth-form.active').appendChild(err);
    }
    err.textContent = msg;
    setTimeout(() => err && err.remove(), 3000);
}

const shakeStyle = document.createElement('style');
shakeStyle.textContent = `@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-8px)}40%{transform:translateX(8px)}60%{transform:translateX(-6px)}80%{transform:translateX(6px)}}`;
document.head.appendChild(shakeStyle);

/* ═══════════════════════════════════════════
   CAT MOOD IMAGES
═══════════════════════════════════════════ */
const catMoodConfig = {
    happy:   { emoji: '😺', glow: '#34d399', tag: 'cute' },
    sad:     { emoji: '😿', glow: '#60a5fa', tag: 'sad' },
    angry:   { emoji: '😾', glow: '#f87171', tag: 'grumpy' },
    fearful: { emoji: '🙀', glow: '#fbbf24', tag: 'scared' },
    fear:    { emoji: '🙀', glow: '#fbbf24', tag: 'scared' }
};

function loadCatImage(emotion) {
    const cfg = catMoodConfig[emotion] || catMoodConfig.happy;
    const img = document.getElementById('cat-mood-img');
    const glow = document.getElementById('cat-mood-glow');
    const ts = Date.now();
    img.src = `https://cataas.com/cat/${cfg.tag}?t=${ts}`;
    img.onerror = () => { img.src = `https://cataas.com/cat?t=${ts}`; };
    glow.style.background = cfg.glow;
    document.getElementById('result-emoji').textContent = cfg.emoji;
}

/* ═══════════════════════════════════════════
   PREDICTION HISTORY
═══════════════════════════════════════════ */
let predHistory = JSON.parse(sessionStorage.getItem('mm_history') || '[]');

function saveToHistory(entry) {
    predHistory.unshift(entry);
    sessionStorage.setItem('mm_history', JSON.stringify(predHistory));
    if (currentMode === 'history') renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;
    if (predHistory.length === 0) {
        list.innerHTML = '<p class="no-history">No predictions yet. Analyze something first!</p>';
        return;
    }
    list.innerHTML = predHistory.map((h, i) => `
        <div class="history-item">
            <div class="hist-left">
                <span class="hist-emoji">${catMoodConfig[h.emotion]?.emoji || '🐱'}</span>
                <div>
                    <div class="hist-emotion">${h.emotion}</div>
                    <div class="hist-meta">${h.mode} · ${h.confidence}% · ${h.time}</div>
                </div>
            </div>
            <button class="hist-del" onclick="deleteHistory(${i})">✕</button>
        </div>
    `).join('');
}

function deleteHistory(i) {
    predHistory.splice(i, 1);
    sessionStorage.setItem('mm_history', JSON.stringify(predHistory));
    renderHistory();
}

function clearHistory() {
    predHistory = [];
    sessionStorage.removeItem('mm_history');
    renderHistory();
    showToast('History cleared', 'info');
}

function exportHistory(format) {
    if (predHistory.length === 0) { showToast('No history to export', 'warn'); return; }
    let content, filename, mime;
    if (format === 'json') {
        content = JSON.stringify(predHistory, null, 2);
        filename = 'meowmood_history.json';
        mime = 'application/json';
    } else {
        const rows = [['Emotion','Confidence','Mode','Time']].concat(predHistory.map(h => [h.emotion, h.confidence, h.mode, h.time]));
        content = rows.map(r => r.join(',')).join('\n');
        filename = 'meowmood_history.csv';
        mime = 'text/csv';
    }
    const blob = new Blob([content], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    showToast(`Exported as ${format.toUpperCase()}`, 'success');
}

/* ═══════════════════════════════════════════
   MIC RECORDING
═══════════════════════════════════════════ */
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

async function toggleRecording() {
    const btn = document.getElementById('mic-btn');
    const status = document.getElementById('mic-status');
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recordedChunks = [];
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'audio/wav' });
                const file = new File([blob], 'recorded_meow.wav', { type: 'audio/wav' });
                stream.getTracks().forEach(t => t.stop());
                handleFile(file);
                status.textContent = '';
                btn.textContent = '🎤 Record Meow';
                btn.classList.remove('recording');
            };
            mediaRecorder.start();
            isRecording = true;
            btn.textContent = '⏹ Stop Recording';
            btn.classList.add('recording');
            status.textContent = '● Recording...';
        } catch (err) {
            showToast('Microphone access denied', 'error');
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
    }
}

/* ═══════════════════════════════════════════
   FUSION TAB FILES
═══════════════════════════════════════════ */
let fusionImageFile = null;
let fusionAudioFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const fii = document.getElementById('fusion-image-input');
    const fai = document.getElementById('fusion-audio-input');
    const dzfi = document.getElementById('drop-zone-fusion-image');
    const dzfa = document.getElementById('drop-zone-fusion-audio');

    if (dzfi) dzfi.onclick = () => fii.click();
    if (dzfa) dzfa.onclick = () => fai.click();
    if (fii) fii.onchange = e => handleFusionFile('image', e.target.files[0]);
    if (fai) fai.onchange = e => handleFusionFile('audio', e.target.files[0]);

    [dzfi, dzfa].forEach(zone => {
        if (!zone) return;
        zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--primary)'; });
        zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
        zone.addEventListener('drop', e => {
            e.preventDefault(); zone.style.borderColor = '';
            const file = e.dataTransfer.files[0];
            const type = zone.id.includes('image') ? 'image' : 'audio';
            if (file) handleFusionFile(type, file);
        });
    });
});

function handleFusionFile(type, file) {
    if (!file) return;
    if (type === 'image') {
        fusionImageFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('fusion-img-output').src = e.target.result;
            document.getElementById('drop-zone-fusion-image').classList.add('hidden');
            document.getElementById('fusion-image-preview').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        fusionAudioFile = file;
        document.getElementById('fusion-audio-name').textContent = file.name;
        document.getElementById('drop-zone-fusion-audio').classList.add('hidden');
        document.getElementById('fusion-audio-preview').classList.remove('hidden');
    }
    updateAnalyzeBtn();
}

function clearFusionFile(type) {
    if (type === 'image') {
        fusionImageFile = null;
        document.getElementById('drop-zone-fusion-image').classList.remove('hidden');
        document.getElementById('fusion-image-preview').classList.add('hidden');
        document.getElementById('fusion-image-input').value = '';
    } else {
        fusionAudioFile = null;
        document.getElementById('drop-zone-fusion-audio').classList.remove('hidden');
        document.getElementById('fusion-audio-preview').classList.add('hidden');
        document.getElementById('fusion-audio-input').value = '';
    }
    updateAnalyzeBtn();
}

/* ═══════════════════════════════════════════
   MAIN APP LOGIC
═══════════════════════════════════════════ */
let currentMode = 'image';
let selectedFile = null;

const dropZoneImage = document.getElementById('drop-zone-image');
const dropZoneAudio = document.getElementById('drop-zone-audio');
const imageInput    = document.getElementById('image-input');
const audioInput    = document.getElementById('audio-input');
const analyzeBtn    = document.getElementById('analyze-btn');

function switchTab(mode, e) {
    currentMode = mode;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    if (e && e.target) e.target.classList.add('active');
    document.getElementById(`${mode}-section`).classList.add('active');

    const isHistory = mode === 'history';
    analyzeBtn.style.display = isHistory ? 'none' : 'flex';
    document.getElementById('result-container').classList.add('hidden');

    if (isHistory) { renderHistory(); return; }

    if (mode !== 'fusion') { selectedFile = null; }
    updateAnalyzeBtn();
}

function updateAnalyzeBtn() {
    if (currentMode === 'fusion') {
        analyzeBtn.disabled = !(fusionImageFile && fusionAudioFile);
    } else {
        analyzeBtn.disabled = !selectedFile;
    }
}

dropZoneImage.onclick = () => imageInput.click();
dropZoneAudio.onclick = () => audioInput.click();

[dropZoneImage, dropZoneAudio].forEach(zone => {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--primary)'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
});

imageInput.onchange = e => handleFile(e.target.files[0]);
audioInput.onchange = e => handleFile(e.target.files[0]);

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    if (currentMode === 'image') {
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('img-output').src = e.target.result;
            document.getElementById('image-section').querySelector('.upload-area').classList.add('hidden');
            document.getElementById('image-preview').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        document.getElementById('audio-filename').textContent = file.name;
        document.getElementById('audio-section').querySelector('.upload-area').classList.add('hidden');
        document.getElementById('audio-preview').classList.remove('hidden');
    }
    updateAnalyzeBtn();
}

function clearPreview(mode) {
    selectedFile = null;
    stopScan(mode);
    document.getElementById(`${mode}-section`).querySelector('.upload-area').classList.remove('hidden');
    document.getElementById(`${mode}-preview`).classList.add('hidden');
    document.getElementById(`${mode}-input`).value = '';
    updateAnalyzeBtn();
    document.getElementById('result-container').classList.add('hidden');
}

function startScan(mode) {
    if (mode === 'image' || mode === 'fusion') {
        document.getElementById('scan-line')?.classList.add('scanning');
        document.getElementById('scan-overlay')?.classList.add('scanning');
        const img = document.getElementById('img-output');
        if (img) img.style.filter = 'brightness(0.85) saturate(1.2)';
    } else {
        document.getElementById('scan-line-audio')?.classList.add('scanning');
    }
}

function stopScan(mode) {
    document.getElementById('scan-line')?.classList.remove('scanning');
    document.getElementById('scan-overlay')?.classList.remove('scanning');
    document.getElementById('scan-line-audio')?.classList.remove('scanning');
    const img = document.getElementById('img-output');
    if (img) img.style.filter = '';
}

analyzeBtn.onclick = async () => {
    const loader = document.getElementById('loader');
    analyzeBtn.disabled = true;
    loader.classList.remove('hidden');
    document.getElementById('result-container').classList.add('hidden');
    startScan(currentMode);

    try {
        let data;
        if (currentMode === 'fusion') {
            const fd = new FormData();
            fd.append('image', fusionImageFile);
            fd.append('audio', fusionAudioFile);
            const res = await fetch('/predict_fusion', { method: 'POST', body: fd });
            data = await res.json();
            if (!data.error) {
                const sub = document.getElementById('fusion-sub-results');
                document.getElementById('fusion-img-label').textContent = `📸 Image: ${data.image_prediction}`;
                document.getElementById('fusion-audio-label').textContent = `🎙️ Audio: ${data.audio_prediction}`;
                sub.classList.remove('hidden');
            }
        } else {
            const fd = new FormData();
            fd.append('file', selectedFile);
            const endpoint = currentMode === 'image' ? '/predict_image' : '/predict_audio';
            const res = await fetch(endpoint, { method: 'POST', body: fd });
            data = await res.json();
        }

        if (data.error) {
            showToast(data.error, 'error');
        } else {
            showResults(data);
            saveToHistory({
                emotion: data.prediction,
                confidence: data.confidence,
                mode: currentMode,
                time: new Date().toLocaleTimeString()
            });
        }
    } catch (err) {
        showToast('Server communication error occurred.', 'error');
    } finally {
        stopScan(currentMode);
        loader.classList.add('hidden');
        updateAnalyzeBtn();
    }
};

function showResults(data) {
    const emotion = data.prediction;
    document.getElementById('result-emotion').textContent = emotion;
    document.getElementById('result-confidence').textContent = data.confidence;
    loadCatImage(emotion);

    Object.keys(data.all_scores).forEach(em => {
        const bar = document.getElementById(`bar-${em}`);
        const pct = document.getElementById(`pct-${em}`);
        if (bar) {
            const val = (data.all_scores[em] * 100).toFixed(0);
            setTimeout(() => { bar.style.width = `${val}%`; }, 100);
            if (pct) pct.textContent = `${val}%`;
        }
    });

    // Show emotion tip
    const tipEl = document.getElementById('emotion-tip');
    if (tipEl && data.tip) {
        tipEl.textContent = data.tip;
        tipEl.classList.remove('hidden');
    }

    document.getElementById('result-container').classList.remove('hidden');
    setTimeout(() => {
        document.getElementById('result-container').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 200);
}

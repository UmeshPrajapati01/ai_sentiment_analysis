import os
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

from database.database_logic import db, init_db, User, Prediction, ModelLog, PLANS, CREDIT_COSTS
from backend.inference.predictor import predict_image, predict_audio

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'front_end' / 'templates'),
    static_folder=str(BASE_DIR / 'front_end' / 'static')
)

app.config['SECRET_KEY'] = 'meowmood_secret_2026'
DB_PATH = BASE_DIR / 'database' / 'cat_emotion.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav'}

init_db(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_stats(user_id):
    # Stats use ModelLog (permanent) so counts never reset when user clears history
    all_preds = ModelLog.query.filter_by(user_id=user_id).all()
    total = len(all_preds)
    image_count = sum(1 for p in all_preds if p.file_type == 'image')
    audio_count = sum(1 for p in all_preds if p.file_type == 'audio')
    emotion_counts = Counter(p.prediction_result for p in all_preds)
    # Recent activity still comes from user-facing Prediction table
    recent = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).limit(10).all()
    return {
        'total_predictions': total,
        'image_count': image_count,
        'audio_count': audio_count,
        'unique_emotions': len(emotion_counts),
        'emotion_distribution': dict(emotion_counts),
        'recent_predictions': recent,
    }

# ── PUBLIC ──────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('landing'))

@app.route('/landing')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html', plans=PLANS)

@app.route('/ppt')
def ppt():
    return send_from_directory(str(BASE_DIR / 'ppt_like_work'), 'index.html')

# ── AUTH ─────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        full_name    = request.form.get('full_name', '').strip()
        email        = request.form.get('email', '').strip()
        password     = request.form.get('password', '')
        confirm_pass = request.form.get('confirm_password', '')
        if not email or not password:
            flash('Email and password are required.', 'danger'); return redirect(url_for('register'))
        if '@' not in email:
            flash('Please enter a valid email address.', 'danger'); return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger'); return redirect(url_for('register'))
        if password != confirm_pass:
            flash('Passwords do not match.', 'danger'); return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning'); return redirect(url_for('register'))
        user = User(
            name=full_name or 'Cat Lover',
            email=email,
            password_hash=bcrypt.generate_password_hash(password).decode('utf-8'),
            credits=20,
            plan='free',
        )
        db.session.add(user); db.session.commit()
        flash('Account created! You get 20 free credits to start. 🎉', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not email or not password:
            flash('Please enter email and password.', 'danger'); return redirect(url_for('login'))
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Welcome back! 🐾', 'success')
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Login unsuccessful. Check email and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out. See you soon! 👋', 'info')
    return redirect(url_for('login'))

# ── DASHBOARD ────────────────────────────────────────────────────────────────

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = None
    credits_used    = 0
    file_type_used  = None
    face_detected   = True   # default True — warning only shown when False

    if request.method == 'POST':
        file      = request.files.get('file')
        file_type = request.form.get('file_type', 'image')

        if not file or file.filename == '':
            flash('No file selected.', 'danger'); return redirect(request.url)

        # Credit check
        cost = CREDIT_COSTS.get(file_type, 1)
        if current_user.credits < cost:
            flash('credits_exhausted', 'warning')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                if file_type == 'image':
                    result, confidence, face_detected = predict_image(filepath)
                elif file_type == 'audio':
                    result, confidence = predict_audio(filepath)
                else:
                    result, confidence = 'Unknown', 0.0

                # Deduct credits
                credits_used   = current_user.deduct_credits(file_type)
                file_type_used = file_type
                db.session.commit()

                prediction_text = result
                pred_entry = Prediction(
                    user_id=current_user.id,
                    file_type=file_type,
                    filename=filename,
                    prediction_result=result,
                    confidence=confidence,
                    credits_used=credits_used,
                )
                db.session.add(pred_entry)
                # ModelLog — permanent, never deleted, powers charts
                db.session.add(ModelLog(
                    user_id=current_user.id,
                    file_type=file_type,
                    prediction_result=result,
                    confidence=confidence,
                ))
                db.session.commit()
                flash(f'credit_used:{credits_used}:{file_type}', 'info')

            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
        else:
            flash('Invalid file type. Allowed: PNG, JPG, JPEG, GIF, MP3, WAV', 'warning')

    stats = get_user_stats(current_user.id)
    return render_template('dashboard.html', prediction=prediction_text,
                           credits_used=credits_used, file_type_used=file_type_used,
                           face_detected=face_detected,
                           plans=PLANS, credit_costs=CREDIT_COSTS, **stats)

# ── FUSION (AJAX) ─────────────────────────────────────────────────────────────

@app.route('/fusion', methods=['POST'])
@login_required
def fusion():
    image_file = request.files.get('image_file')
    audio_file = request.files.get('audio_file')
    if not image_file or not audio_file:
        return jsonify({'error': 'Both image and audio files are required.'}), 400

    cost = CREDIT_COSTS.get('fusion', 3)
    if current_user.credits < cost:
        return jsonify({'error': 'credits_exhausted', 'credits': current_user.credits}), 402

    try:
        img_filename = secure_filename(image_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        image_file.save(img_path)

        aud_filename = secure_filename(audio_file.filename)
        aud_path = os.path.join(app.config['UPLOAD_FOLDER'], aud_filename)
        audio_file.save(aud_path)

        img_result, img_conf, _ = predict_image(img_path)
        aud_result, aud_conf = predict_audio(aud_path)
        fusion_result = img_result if img_result.lower() == aud_result.lower() else f"{img_result} / {aud_result}"
        fusion_conf = round((img_conf + aud_conf) / 2, 2)

        current_user.deduct_credits('fusion')
        db.session.commit()

        pred_entry = Prediction(
            user_id=current_user.id,
            file_type='fusion',
            filename=f"{img_filename} + {aud_filename}",
            prediction_result=fusion_result,
            confidence=fusion_conf,
            credits_used=cost,
        )
        db.session.add(pred_entry)
        # ModelLog — permanent, never deleted, powers charts
        db.session.add(ModelLog(
            user_id=current_user.id,
            file_type='fusion',
            prediction_result=fusion_result,
            confidence=fusion_conf,
        ))
        db.session.commit()

        return jsonify({
            'image': img_result,
            'audio': aud_result,
            'fusion': fusion_result,
            'credits_used': cost,
            'credits_remaining': current_user.credits,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── SUBSCRIPTION ──────────────────────────────────────────────────────────────

@app.route('/plans')
@login_required
def plans():
    return render_template('plans.html', plans=PLANS, credit_costs=CREDIT_COSTS)

@app.route('/subscribe/<plan_key>', methods=['POST'])
@login_required
def subscribe(plan_key):
    if plan_key not in PLANS:
        flash('Invalid plan.', 'danger'); return redirect(url_for('plans'))
    plan = PLANS[plan_key]
    current_user.plan    = plan_key
    current_user.credits = plan['credits']
    db.session.commit()
    flash(f'🎉 Subscribed to {plan["name"]} plan! {plan["credits"]} credits added.', 'success')
    return redirect(url_for('dashboard'))

# ── OTHER PAGES ───────────────────────────────────────────────────────────────

@app.route('/profile')
@login_required
def profile():
    stats = get_user_stats(current_user.id)
    return render_template('profile.html', plans=PLANS, **stats)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    display_name = request.form.get('display_name', '').strip()
    if display_name:
        current_user.name = display_name
        db.session.commit()
        flash('Profile updated successfully! 🎉', 'success')
    else:
        flash('Display name cannot be empty.', 'danger')
    return redirect(url_for('profile'))

@app.route('/metrics')
@login_required
def metrics():
    stats = get_user_stats(current_user.id)
    return render_template('metrics.html', **stats)

@app.route('/history')
@login_required
def history():
    user_history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=user_history)

@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('History cleared successfully. 🗑️', 'success')
    return redirect(url_for('history'))

@app.route('/history/delete/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.filter_by(id=pred_id, user_id=current_user.id).first()
    if pred:
        db.session.delete(pred)
        db.session.commit()
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 404

@app.route('/history/download')
@login_required
def download_history():
    preds = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    lines = ['#,Type,Filename,Emotion Detected,Credits Used,Date & Time']
    for i, p in enumerate(preds, 1):
        ts = p.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        filename_safe = p.filename.replace(',', ';')
        result_safe   = p.prediction_result.replace(',', ';')
        lines.append(f'{i},{p.file_type},{filename_safe},{result_safe},{p.credits_used},{ts}')
    csv_data = '\n'.join(lines)
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=meowmood_history.csv'}
    )

@app.route('/model-history')
@login_required
def model_history():
    logs = ModelLog.query.filter_by(user_id=current_user.id).order_by(ModelLog.timestamp.desc()).all()
    return render_template('model_history.html', logs=logs)

@app.route('/model-history/download')
@login_required
def download_model_history():
    logs = ModelLog.query.filter_by(user_id=current_user.id).order_by(ModelLog.timestamp.desc()).all()
    lines = ['#,Type,Emotion Detected,Confidence,Date & Time']
    for i, l in enumerate(logs, 1):
        ts   = l.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        conf = f'{l.confidence:.2f}' if l.confidence else 'N/A'
        lines.append(f'{i},{l.file_type},{l.prediction_result},{conf},{ts}')
    csv_data = '\n'.join(lines)
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=meowmood_model_history.csv'}
    )

@app.route('/report')
@login_required
def model_report():
    import re

    def parse_report(filepath):
        """Parse a sklearn classification_report text into structured data."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            text = f.read()
        rows = []
        accuracy = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # accuracy line
            acc_match = re.match(r'accuracy\s+([\d.]+)\s+(\d+)', line)
            if acc_match:
                accuracy = float(acc_match.group(1))
                continue
            # class rows: name  prec  recall  f1  support
            parts = line.split()
            if len(parts) >= 5:
                try:
                    support = int(parts[-1])
                    f1      = float(parts[-2])
                    recall  = float(parts[-3])
                    prec    = float(parts[-4])
                    name    = ' '.join(parts[:-4])
                    if name not in ('macro avg', 'weighted avg', 'accuracy'):
                        rows.append({'name': name, 'precision': prec, 'recall': recall, 'f1': f1, 'support': support})
                except (ValueError, IndexError):
                    pass
        return {'rows': rows, 'accuracy': accuracy}

    MODEL_DIR = BASE_DIR / 'backend' / 'trained_modelimages'
    image_report_path = MODEL_DIR / 'image_model' / 'classification_report.txt'
    audio_report_path = MODEL_DIR / 'audio_model_finetuned' / 'classification_report.txt'

    image_report = parse_report(str(image_report_path))
    audio_report = parse_report(str(audio_report_path))

    # Image training history for chart
    img_history_path = MODEL_DIR / 'image_model_finetuned' / 'training_history.csv'
    img_history = []
    if img_history_path.exists():
        import csv
        with open(img_history_path, 'r') as f:
            reader = csv.DictReader(f)
            img_history = list(reader)

    return render_template('report.html',
                           image_report=image_report,
                           audio_report=audio_report,
                           img_history=img_history)

@app.route('/settings', methods=['GET'])
@login_required
def settings():
    return render_template('settings.html')

@app.route('/settings/save', methods=['POST'])
@login_required
def settings_save():
    section = request.form.get('section')

    if section == 'account':
        display_name = request.form.get('display_name', '').strip()
        if display_name:
            current_user.name = display_name
            db.session.commit()
            flash('Account info updated. 🎉', 'success')
        else:
            flash('Display name cannot be empty.', 'danger')

    elif section == 'password':
        current_pw  = request.form.get('current_password', '')
        new_pw      = request.form.get('new_password', '')
        confirm_pw  = request.form.get('confirm_password', '')
        if not bcrypt.check_password_hash(current_user.password_hash, current_pw):
            flash('Current password is incorrect.', 'danger')
        elif len(new_pw) < 6:
            flash('New password must be at least 6 characters.', 'danger')
        elif new_pw != confirm_pw:
            flash('Passwords do not match.', 'danger')
        else:
            current_user.password_hash = bcrypt.generate_password_hash(new_pw).decode('utf-8')
            db.session.commit()
            flash('Password updated successfully. 🔒', 'success')

    elif section == 'notifications':
        # Stored client-side via localStorage; just acknowledge
        flash('Notification preferences saved. 🔔', 'success')

    return redirect(url_for('settings'))

@app.route('/api/chart-data')
@login_required
def chart_data():
    # Charts always use ModelLog — permanent, unaffected by user history deletion
    all_preds = ModelLog.query.filter_by(user_id=current_user.id).all()
    today = datetime.utcnow().date()

    # Emotion distribution — only single-emotion results (no fusion "X / Y")
    emotion_map = {}
    for p in all_preds:
        raw = p.prediction_result.strip()
        if '/' in raw:
            # split fusion results into individual emotions
            for part in raw.split('/'):
                key = part.strip().capitalize()
                if key:
                    emotion_map[key] = emotion_map.get(key, 0) + 1
        else:
            key = (raw[0].upper() + raw[1:]) if raw else raw
            emotion_map[key] = emotion_map.get(key, 0) + 1

    # Date range: first prediction to today, max 30 days
    if all_preds:
        first_date = min(p.timestamp.date() for p in all_preds)
        start_date = max(first_date, today - timedelta(days=29))
    else:
        start_date = today

    # Always show at least 7 days so the chart has width
    if (today - start_date).days < 6:
        start_date = today - timedelta(days=6)

    num_days = (today - start_date).days + 1
    daily_labels, daily_values, cum_labels, cumulative = [], [], [], []
    daily_image, daily_audio, daily_fusion = [], [], []
    running = sum(1 for p in all_preds if p.timestamp.date() < start_date)  # carry-over
    for i in range(num_days):
        d = start_date + timedelta(days=i)
        day_preds = [p for p in all_preds if p.timestamp.date() == d]
        count = len(day_preds)
        running += count
        label = d.strftime('%b %d')
        daily_labels.append(label)
        daily_values.append(count)
        cum_labels.append(label)
        cumulative.append(running)
        daily_image.append(sum(1 for p in day_preds if p.file_type == 'image'))
        daily_audio.append(sum(1 for p in day_preds if p.file_type == 'audio'))
        daily_fusion.append(sum(1 for p in day_preds if p.file_type == 'fusion'))

    # Hourly breakdown today
    hourly = {h: {'image': 0, 'audio': 0, 'fusion': 0} for h in range(24)}
    for p in all_preds:
        if p.timestamp.date() == today:
            h = p.timestamp.hour
            t = p.file_type if p.file_type in ('image', 'audio', 'fusion') else 'image'
            hourly[h][t] += 1

    return jsonify({
        'emotion_distribution': emotion_map,
        'daily_activity': {'labels': daily_labels, 'values': daily_values, 'image': daily_image, 'audio': daily_audio, 'fusion': daily_fusion},
        'hourly': {
            'labels': [str(h)+':00' for h in range(24)],
            'image':  [hourly[h]['image']  for h in range(24)],
            'audio':  [hourly[h]['audio']  for h in range(24)],
            'fusion': [hourly[h]['fusion'] for h in range(24)],
        },
        'cumulative': {'labels': cum_labels, 'values': cumulative},
        'totals': {
            'total':  len(all_preds),
            'image':  sum(1 for p in all_preds if p.file_type == 'image'),
            'audio':  sum(1 for p in all_preds if p.file_type == 'audio'),
            'fusion': sum(1 for p in all_preds if p.file_type == 'fusion'),
        }
    })


@app.route('/media/<path:filename>')
def serve_media(filename):
    return send_from_directory(str(BASE_DIR / 'front_end' / 'images_videos'), filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(str(BASE_DIR / 'front_end' / 'front_end_png'), filename)

if __name__ == '__main__':
    print("=" * 60)
    print("  🐱 MeowMood - AI Cat Emotion Recognition")
    print("  🌐 http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)

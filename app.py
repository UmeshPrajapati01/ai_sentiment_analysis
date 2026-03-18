
import os
from pathlib import Path
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

# Import local modules
from database.database_logic import db, init_db, User, Prediction
from backend.inference.predictor import predict_image, predict_audio

# ==========================================
# APP CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'front_end' / 'templates'),
    static_folder=str(BASE_DIR / 'front_end' / 'static')
)

app.config['SECRET_KEY'] = 'your_secret_key_here_change_in_production'

# Database Configuration (SQLite)
DB_PATH = BASE_DIR / 'database' / 'cat_emotion.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File Upload Configuration
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav'}

# ==========================================
# INITIALIZE EXTENSIONS
# ==========================================
init_db(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# HELPER: Get user stats
# ==========================================
def get_user_stats(user_id):
    """Get prediction statistics for a user."""
    all_preds = Prediction.query.filter_by(user_id=user_id).all()
    total = len(all_preds)
    image_count = sum(1 for p in all_preds if p.file_type == 'image')
    audio_count = sum(1 for p in all_preds if p.file_type == 'audio')

    # Emotion distribution
    emotion_counts = Counter(p.prediction_result for p in all_preds)
    unique_emotions = len(emotion_counts)

    # Recent predictions (last 10)
    recent = Prediction.query.filter_by(user_id=user_id) \
        .order_by(Prediction.timestamp.desc()).limit(10).all()

    return {
        'total_predictions': total,
        'image_count': image_count,
        'audio_count': audio_count,
        'unique_emotions': unique_emotions,
        'emotion_distribution': dict(emotion_counts),
        'recent_predictions': recent
    }

# ==========================================
# ROUTES: PUBLIC PAGES
# ==========================================

@app.route('/')
def home():
    """Root route - redirect to dashboard if logged in, else to landing."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('landing'))

@app.route('/landing')
def landing():
    """Landing / home page for non-authenticated users."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# ==========================================
# ROUTE: PPT PRESENTATION
# ==========================================

@app.route('/ppt')
def ppt():
    """Serve the PPT-like presentation from ppt_like_work/index.html."""
    ppt_dir = BASE_DIR / 'ppt_like_work'
    return send_from_directory(str(ppt_dir), 'index.html')

# ==========================================
# ROUTES: AUTH (Register, Login, Logout)
# ==========================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_pass = request.form.get('confirm_password', '')

        # Validation
        if not email or not password:
            flash('Email and password are required.', 'danger')
            return redirect(url_for('register'))

        if '@' not in email:
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_pass:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('register'))

        # Create User
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(
            name=full_name if full_name else 'Cat Lover',
            email=email,
            password_hash=hashed_password
        )
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully! You can now login. 🎉', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter email and password.', 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful! Welcome back! 🐾', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out. See you soon! 👋', 'info')
    return redirect(url_for('login'))

# ==========================================
# ROUTES: AUTHENTICATED PAGES
# ==========================================

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in request.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        file_type = request.form.get('file_type', 'image')  # 'image' or 'audio'

        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform Prediction
            try:
                if file_type == 'image':
                    result = predict_image(filepath)
                elif file_type == 'audio':
                    result = predict_audio(filepath)
                else:
                    result = "Unknown Type"

                prediction_text = result

                # Save to History
                pred_entry = Prediction(
                    user_id=current_user.id,
                    file_type=file_type,
                    filename=filename,
                    prediction_result=result
                )
                db.session.add(pred_entry)
                db.session.commit()

                flash(f'File processed successfully! Emotion detected: {result} 🎉', 'success')

            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
        else:
            flash('Invalid file type. Allowed: PNG, JPG, JPEG, GIF, MP3, WAV', 'warning')

    # Get user stats
    stats = get_user_stats(current_user.id)

    return render_template(
        'dashboard.html',
        prediction=prediction_text,
        **stats
    )

@app.route('/profile')
@login_required
def profile():
    stats = get_user_stats(current_user.id)
    return render_template('profile.html', **stats)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    # For now, just redirect back with a success message
    flash('Profile updated successfully! 🎉', 'success')
    return redirect(url_for('profile'))

@app.route('/metrics')
@login_required
def metrics():
    stats = get_user_stats(current_user.id)
    return render_template('metrics.html', **stats)

@app.route('/history')
@login_required
def history():
    user_history = Prediction.query.filter_by(user_id=current_user.id) \
        .order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=user_history)

# ==========================================
# ROUTES: STATIC ASSET SERVING
# ==========================================

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve videos and images from front_end/images_videos."""
    media_dir = BASE_DIR / 'front_end' / 'images_videos'
    return send_from_directory(str(media_dir), filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve assets from front_end/front_end_png."""
    return send_from_directory(str(BASE_DIR / 'front_end' / 'front_end_png'), filename)

# ==========================================
# RUN APP
# ==========================================
if __name__ == '__main__':
    print("============================================================")
    print("  MewConnect - Cat Emotion Recognition System")
    print("  Starting at http://127.0.0.1:5000")
    print("============================================================")
    app.run(debug=True, port=5000)

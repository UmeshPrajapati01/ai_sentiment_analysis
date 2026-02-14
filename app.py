
import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash # Import explicitly for clarity/safety if Bcrypt fails

# Import local modules
from database.database_logic import db, init_db, User, Prediction
from backend.inference.predictor import predict_image, predict_audio

# Configuration
app = Flask(__name__, template_folder='front_end/templates', static_folder='front_end/static')
app.config['SECRET_KEY'] = 'your_secret_key_here_change_in_production'

# Database Configuration (SQLite)
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'database' / 'cat_emotion.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File Upload Configuration
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav'}

# Initialize Extensions
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
# ROUTES
# ==========================================

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_pass = request.form.get('confirm_password')

        # Validation
        if not email.endswith('@gmail.com'):
            flash('Only Gmail addresses are allowed.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_pass:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('register'))

        # Create User
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created! You can now login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        file_type = request.form.get('file_type') # 'image' or 'audio'
        
        if file.filename == '':
            flash('No selected file', 'danger')
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
                
                flash('File processed successfully!', 'success')
                
            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
        else:
            flash('Invalid file type.', 'warning')
            
    return render_template('dashboard.html', prediction=prediction_text)

@app.route('/history')
@login_required
def history():
    user_history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', history=user_history)

# Helper route to serve assets from front_end folder
@app.route('/assets/<filename>')
def get_asset(filename):
    return send_from_directory('front_end/front_end_png', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

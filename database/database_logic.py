
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True, default='Cat Lover')
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    @property
    def display_name(self):
        """Return user's name, or derive from email if not set."""
        if self.name and self.name != 'Cat Lover':
            return self.name
        return self.email.split('@')[0].capitalize()

    def __repr__(self):
        return f"User('{self.name}', '{self.email}')"

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False) # 'image' or 'audio'
    filename = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True) # Optional
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Prediction('{self.filename}', '{self.prediction_result}')"

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

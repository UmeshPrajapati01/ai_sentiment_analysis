from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Credit costs per analysis type
CREDIT_COSTS = {
    'image':  1,
    'audio':  2,
    'fusion': 3,
}

# Subscription plans
PLANS = {
    'free':    {'name': 'Free',    'credits': 20,   'price': 0,    'period': None},
    'weekly':  {'name': 'Weekly',  'credits': 100,  'price': 2.99, 'period': 'week'},
    'monthly': {'name': 'Monthly', 'credits': 500,  'price': 7.99, 'period': 'month'},
    'yearly':  {'name': 'Yearly',  'credits': 9999, 'price': 49.99,'period': 'year'},
}


class User(db.Model, UserMixin):
    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(100), nullable=True, default='Cat Lover')
    email           = db.Column(db.String(120), unique=True, nullable=False)
    password_hash   = db.Column(db.String(128), nullable=False)

    # Credits & subscription
    credits         = db.Column(db.Integer, default=20, nullable=False)
    plan            = db.Column(db.String(20), default='free', nullable=False)
    plan_expires_at = db.Column(db.DateTime, nullable=True)

    predictions = db.relationship('Prediction', backref='author', lazy=True)

    @property
    def display_name(self):
        if self.name and self.name != 'Cat Lover':
            return self.name
        return self.email.split('@')[0].capitalize()

    @property
    def plan_label(self):
        return PLANS.get(self.plan, PLANS['free'])['name']

    def has_credits(self, analysis_type='image'):
        cost = CREDIT_COSTS.get(analysis_type, 1)
        return self.credits >= cost

    def deduct_credits(self, analysis_type='image'):
        cost = CREDIT_COSTS.get(analysis_type, 1)
        self.credits = max(0, self.credits - cost)
        return cost

    def __repr__(self):
        return f"User('{self.name}', '{self.email}', credits={self.credits}, plan={self.plan})"


class Prediction(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type         = db.Column(db.String(10), nullable=False)
    filename          = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence        = db.Column(db.Float, nullable=True)
    credits_used      = db.Column(db.Integer, default=1, nullable=False)
    timestamp         = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Prediction('{self.filename}', '{self.prediction_result}')"


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _seed_premium_account(app)


def _seed_premium_account(app):
    """Ensure umesh.prajapati0506@gmail.com has yearly premium + unlimited credits."""
    from flask_bcrypt import Bcrypt
    bcrypt = Bcrypt(app)
    email = 'umesh.prajapati0506@gmail.com'
    user = User.query.filter_by(email=email).first()
    if user:
        user.plan = 'yearly'
        user.credits = 9999
        user.plan_expires_at = datetime(2027, 1, 1)
    else:
        user = User(
            name='Umesh Prajapati',
            email=email,
            password_hash=bcrypt.generate_password_hash('Umesh@2026').decode('utf-8'),
            plan='yearly',
            credits=9999,
            plan_expires_at=datetime(2027, 1, 1),
        )
        db.session.add(user)
    db.session.commit()

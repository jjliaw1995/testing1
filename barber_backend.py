from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///barbershop.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Models
class Barber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(80), nullable=False)

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    gender = db.Column(db.String(10))
    first_visit = db.Column(db.Date)
    last_visit = db.Column(db.Date)
    password = db.Column(db.String(80), nullable=False)
    barber_id = db.Column(db.Integer, db.ForeignKey('barber.id'))
    barber = db.relationship('Barber', backref=db.backref('customers', lazy=True))

class ServiceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'))
    service_type = db.Column(db.String(80))
    description = db.Column(db.String(200))
    image_path = db.Column(db.String(200))
    amount_paid = db.Column(db.Float)
    service_date = db.Column(db.DateTime, default=datetime.utcnow)
    remarks = db.Column(db.String(200))
    customer = db.relationship('Customer', backref=db.backref('services', lazy=True))

# Routes
@app.route('/')
def index():
    return redirect(url_for('barber_login'))

# Barber login
@app.route('/barber/login', methods=['GET', 'POST'])
def barber_login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        barber = Barber.query.filter_by(name=name, password=password).first()
        if barber:
            return redirect(url_for('main_page', barber_id=barber.id))
    return render_template_string('''
        <h2>Barber Login</h2>
        <form method="post">
            <input name="name" placeholder="Name">
            <input name="password" placeholder="Password" type="password">
            <button type="submit">Login</button>
        </form>
    ''')

# Main page - customer login
@app.route('/barber/<int:barber_id>/main', methods=['GET', 'POST'])
def main_page(barber_id):
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        customer = Customer.query.filter_by(phone=phone, password=password, barber_id=barber_id).first()
        if customer:
            return redirect(url_for('customer_page', customer_id=customer.id))
    return render_template_string('''
        <h2>Customer Login</h2>
        <form method="post">
            <input name="phone" placeholder="Phone">
            <input name="password" placeholder="Password" type="password">
            <button type="submit">Login</button>
        </form>
    ''')

# Display customer info and service history
@app.route('/customer/<int:customer_id>')
def customer_page(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    services = ServiceRecord.query.filter_by(customer_id=customer_id).order_by(ServiceRecord.service_date.desc()).all()
    return render_template_string('''
        <h2>{{ c.name }} ({{ c.gender }})</h2>
        <p>Phone: {{ c.phone }}</p>
        <p>First visit: {{ c.first_visit }}</p>
        <p>Last visit: {{ c.last_visit }}</p>
        <a href="{{ url_for('add_service', customer_id=c.id) }}">Add Service</a>
        <hr>
        {% for s in services %}
            <div style="border:1px solid #ccc;padding:10px;margin-bottom:10px;">
                <p>{{ s.service_date }} - {{ s.service_type }} - ${{ '{:.2f}'.format(s.amount_paid or 0) }}</p>
                <p>{{ s.description }}</p>
                {% if s.image_path %}
                    <img src="{{ url_for('uploaded_file', filename=s.image_path) }}" style="max-width:200px" onclick="this.style.maxWidth=this.style.maxWidth=='200px'?'600px':'200px';">
                {% endif %}
                <p>{{ s.remarks }}</p>
            </div>
        {% endfor %}
    ''', c=customer, services=services)

# File serving for uploaded images
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Add service record
@app.route('/customer/<int:customer_id>/add', methods=['GET', 'POST'])
def add_service(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    if request.method == 'POST':
        service_type = request.form['service_type']
        description = request.form.get('description')
        amount_paid = request.form.get('amount_paid', type=float)
        remarks = request.form.get('remarks')
        image = request.files.get('image')
        image_path = None
        if image and image.filename:
            filename = f"{datetime.utcnow().timestamp()}_{image.filename}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(path)
            image_path = filename
        record = ServiceRecord(customer_id=customer_id,
                               service_type=service_type,
                               description=description,
                               amount_paid=amount_paid,
                               remarks=remarks,
                               image_path=image_path,
                               service_date=datetime.utcnow())
        db.session.add(record)
        customer.last_visit = datetime.utcnow().date()
        db.session.commit()
        return redirect(url_for('customer_page', customer_id=customer_id))
    return render_template_string('''
        <h2>Add Service for {{ c.name }}</h2>
        <form method="post" enctype="multipart/form-data">
            <input name="service_type" placeholder="Service Type"><br>
            <input name="description" placeholder="Description"><br>
            <input name="amount_paid" placeholder="Amount Paid" type="number" step="0.01"><br>
            <input name="remarks" placeholder="Remarks"><br>
            <input type="file" name="image"><br>
            <button type="submit">Add</button>
        </form>
    ''', c=customer)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


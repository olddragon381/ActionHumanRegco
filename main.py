import cap
from flask import Flask, Response, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import tensorflow
import cv2
import numpy as np

from compute import compute_fourier_features_extended, compute_pose_features, compute_optical_flow_features,apply_ema
import os
from scipy import stats
from collections import deque

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'avi', 'mp4', 'mov'}  # Allowed file types


db = SQLAlchemy(app)


rf_model = joblib.load('model/rf_model2.pkl')
svm_model = joblib.load('model/svm_model2.pkl')
# Mô hình CNN
cnn_model = tensorflow.keras.models.load_model('model/cnn_model2.h5')

# Mô hình CNN + LSTM
cnn_lstm_model = tensorflow.keras.models.load_model('model/cnn_lstm_model2.h5')


# Initialize MediaPipe Pose
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# # Camera setup
# cap = cv2.VideoCapture(0)



# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        # Kiểm tra người dùng đã tồn tại
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('User already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            flash('Login successful!', 'success')
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('home'))
    return render_template('login.html')




@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    else:
        flash('You need to log in first.', 'danger')
        return redirect(url_for('home'))



# Hàm trích xuất tất cả các đặc trưng
def extract_all_features(ema_landmarks1,frame):
    # Fourier features
    fourier_features = compute_fourier_features_extended(frame)
    alpha=0.2
    # Pose features
    pose_features = compute_pose_features(frame)
    ema_landmarks1 = apply_ema(ema_landmarks1, pose_features, alpha)
    # Optical flow features (sử dụng EMA để mượt mà các giá trị)
    optical_flow_features = np.zeros(5)  # Khởi tạo mảng với 5 đặc trưng
    prev_frame = None  # Khởi tạo cho frame đầu tiên

    if prev_frame is not None:
        optical_flow_features = compute_optical_flow_features(prev_frame, frame)

    # Trả về tất cả các đặc trưng đã trích xuất
    return ema_landmarks1,np.concatenate([fourier_features, ema_landmarks1, optical_flow_features])


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    # Dự đoán cho tất cả các frame của từng mô hình
    rf_predictions = deque(maxlen=30)  # Dự đoán RF cho mỗi frame, lưu trữ tối đa 30 frame
    svm_predictions = deque(maxlen=30)  # Dự đoán SVM cho mỗi frame, lưu trữ tối đa 30 frame
    cnn_predictions = deque(maxlen=30)  # Dự đoán CNN cho mỗi frame, lưu trữ tối đa 30 frame
    cnn_lstm_predictions = deque(maxlen=30)  # Dự đoán CNN-LSTM cho mỗi frame, lưu trữ tối đa 30 frame

    ema_landmarks = None
    ema_landmarks1 = None
    alpha = 0
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Trích xuất đặc trưng cho từng frame
        pose_features = compute_pose_features(frame)  # Trích xuất Pose features
        ema_landmarks1, all_features = extract_all_features(ema_landmarks1,frame)  # Trích xuất tất cả các đặc trưng
        ema_landmarks = apply_ema(ema_landmarks, pose_features, alpha)

        print(all_features.shape)
        all_features = all_features.reshape(1,all_features.shape[0])
        # Dự đoán bằng RF và SVM (chỉ sử dụng Pose features)
        rf_prediction = rf_model.predict([ema_landmarks])[0]
        svm_prediction = svm_model.predict([ema_landmarks])[0]

        # Dự đoán bằng CNN và CNN-LSTM (dùng tất cả các đặc trưng)
        cnn_prediction = cnn_model.predict(np.expand_dims(all_features, axis=0))[0]
        cnn_lstm_prediction = cnn_lstm_model.predict(np.expand_dims(all_features, axis=0))[0]

        # Lưu dự đoán cho từng mô hình
        rf_predictions.append(rf_prediction)
        svm_predictions.append(svm_prediction)
        cnn_predictions.append(np.argmax(cnn_prediction))  # Chọn lớp có xác suất cao nhất
        cnn_lstm_predictions.append(np.argmax(cnn_lstm_prediction))  # Chọn lớp có xác suất cao nhất

        frame_idx += 1

    cap.release()

    # Tính Mode cho các dự đoán
    rf_final_prediction = int(stats.mode(rf_predictions)[0])  # Mode của các dự đoán RF
    svm_final_prediction = int(stats.mode(svm_predictions)[0])  # Mode của các dự đoán SVM
    cnn_final_prediction = int(stats.mode(cnn_predictions)[0])  # Mode của các dự đoán CNN
    cnn_lstm_final_prediction = int(stats.mode(cnn_lstm_predictions)[0])  # Mode của các dự đoán CNN-LSTM

    # Kết quả cuối cùng
    result = {
        "RF": rf_final_prediction,
        "SVM": svm_final_prediction,
        "CNN": cnn_final_prediction,
        "CNN_LSTM": cnn_lstm_final_prediction
    }

    # Cleanup
    os.remove(video_path)

    return jsonify(result)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))







# Video streaming function


if __name__ == '__main__':
    app.run(debug=True)





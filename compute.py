
import cv2
import numpy as np
# Initialize MediaPipe Pose
import mediapipe as mp
# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
def compute_fourier_features_extended(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Magnitude Spectrum
    magnitude_spectrum = np.abs(f_shift)
    log_magnitude = 20 * np.log(magnitude_spectrum + 1)

    # Thống kê cơ bản
    mean_val = np.mean(log_magnitude)
    max_val = np.max(log_magnitude)
    min_val = np.min(log_magnitude)
    var_val = np.var(log_magnitude)
    skew_val = np.mean((log_magnitude - mean_val)**3) / (np.var(log_magnitude)**1.5)
    kurtosis_val = np.mean((log_magnitude - mean_val)**4) / (np.var(log_magnitude)**2)
    energy_val = np.sum(log_magnitude ** 2)

    # Phân vùng phổ
    center = log_magnitude.shape[0] // 2
    low_region = log_magnitude[center - 10:center + 10, center - 10:center + 10]
    mid_region = log_magnitude[center - 30:center + 30, center - 30:center + 30]
    high_region = log_magnitude

    low_mean = np.mean(low_region)
    mid_mean = np.mean(mid_region)
    high_mean = np.mean(high_region)

    # Kết hợp tất cả các đặc trưng
    features = [
        mean_val, max_val, min_val, var_val, skew_val, kurtosis_val,
        energy_val, low_mean, mid_mean, high_mean
    ]
    return features



def preprocess_frame_for_pose(image):
    """Preprocess the frame to make it square and resize to fit model input."""
    height, width, _ = image.shape
    # Make the frame square
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    cropped_frame = image[start_y:start_y+size, start_x:start_x+size]
    # Resize to 256x256 (or required size for MediaPipe)
    resized_frame = cv2.resize(cropped_frame, (256, 256))
    return resized_frame

# Hàm trích xuất đặc trưng khung xương từ MediaPipe
def compute_pose_features(image):
    """
    Extract pose features from a single frame using MediaPipe Pose.
    Arguments:
        image: Input image (BGR format).
    Returns:
        Flattened landmark coordinates or zeros if no landmarks are detected.
    """
    # Preprocess frame

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image through MediaPipe Pose
    results = pose.process(rgb_image)

    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
        return np.array(coords).flatten()  # Vectorized (x, y, z) coordinates
    else:
        # Return zeros if no landmarks are detected
        return np.zeros(33 * 3)  # 33 landmarks, each with (x, y, z)

def apply_ema(prev_value, current_value, alpha = 0.2):
    if prev_value is None:
        return current_value  # Nếu khung đầu tiên, gán giá trị hiện tại
    return alpha * current_value + (1 - alpha) * prev_value

def compute_optical_flow_features(prev_frame, curr_frame):
    # Chuyển khung hình về dạng grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Tính toán Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)

    # Tách thành 2 thành phần: độ lớn (magnitude) và góc (angle)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Trích xuất các đặc trưng thống kê từ độ lớn và góc
    mag_mean = np.mean(mag)
    mag_max = np.max(mag)
    mag_var = np.var(mag)

    ang_mean = np.mean(ang)
    ang_var = np.var(ang)

    return [mag_mean, mag_max, mag_var, ang_mean, ang_var]
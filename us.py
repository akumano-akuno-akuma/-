import os
import json
import cv2 #version 4.11.0.86
import math
import numpy as np #version 2.3.3
import torch #version 2.8.0
import smbus
import time
from torch.utils.data import Dataset
from itertools import combinations
import multiprocessing as mp
import subprocess
import shlex

# MPU-6050 I2C 주소 및 레지스터 설정
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45

# I2C 버스 (라즈베리파이 5는 보통 1번)
bus = smbus.SMBus(1)

# MPU-6050 초기화
def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# MPU-6050에서 2바이트 데이터 읽기
def read_word(reg):
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low = bus.read_byte_data(MPU6050_ADDR, reg + 1)
    val = (high << 8) + low
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

# 상보 필터 변수 초기화
pitch = 0.0
roll = 0.0
last_time = time.time()
alpha = 0.98

def get_pitch_roll():
    global pitch, roll, last_time
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # 가속도 데이터 읽기 및 변환
    accel_x = read_word(ACCEL_XOUT_H) / 16384.0
    accel_y = read_word(ACCEL_YOUT_H) / 16384.0
    accel_z = read_word(ACCEL_ZOUT_H) / 16384.0

    # 자이로 데이터 읽기 및 변환
    gyro_x = read_word(GYRO_XOUT_H) / 131.0
    gyro_y = read_word(GYRO_YOUT_H) / 131.0

    # 가속도 센서로 계산한 각도
    pitch_accel = math.atan2(accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180 / math.pi
    roll_accel = math.atan2(accel_y, accel_z) * 180 / math.pi

    # 상보 필터 적용
    pitch = alpha * (pitch + gyro_y * dt) + (1 - alpha) * pitch_accel
    roll = alpha * (roll + gyro_x * dt) + (1 - alpha) * roll_accel

    return pitch, roll

# ====================================
# 항해력 기준별 데이터 예시 (57개 중 일부)
# ====================================
nautical_stars = [
    {"name": "Sirius","ra": 101.287,"dec": -16.716,"vmag": -1.46},
    {"name": "Canopus","ra": 95.987,"dec": -52.695,"vmag": -0.74},
    {"name": "Arcturus","ra": 213.915,"dec": 19.182,"vmag": -0.05},
    {"name": "Vega","ra": 279.234,"dec": 38.783,"vmag": 0.03},
    {"name": "Capella","ra": 79.172,"dec": 45.998,"vmag": 0.08},
    {"name": "Rigel","ra": 77.585,"dec": -8.201,"vmag": 0.13},
    {"name": "Procyon","ra": 114.825,"dec": 5.225,"vmag": 0.34},
    {"name": "Achernar","ra": 22.844,"dec": -57.237,"vmag": 0.45},
    {"name": "Betelgeuse","ra": 88.793,"dec": 7.407,"vmag": 0.50},
    {"name": "Hadar","ra": 210.058,"dec": -60.485,"vmag": 0.61},
    {"name": "Acrux","ra": 185.122,"dec": -63.099,"vmag": 0.77},
    {"name": "Aldebaran","ra": 68.980,"dec": 16.511,"vmag": 0.86},
    {"name": "Spica","ra": 201.298,"dec": -11.162,"vmag": 0.97},
    {"name": "Antares","ra": 247.355,"dec": -26.432,"vmag": 1.06},
    {"name": "Pollux","ra": 116.329,"dec": 28.026,"vmag": 1.16},
    {"name": "Fomalhaut","ra": 344.406,"dec": -29.627,"vmag": 1.17},
    {"name": "Deneb","ra": 310.358,"dec": 45.280,"vmag": 1.25},
    {"name": "Regulus","ra": 152.093,"dec": 11.967,"vmag": 1.35},
    {"name": "Adhara","ra": 105.101,"dec": -28.972,"vmag": 1.50},
    {"name": "Castor","ra": 110.875,"dec": 31.888,"vmag": 1.58},
    {"name": "Gacrux","ra": 182.202,"dec": -57.173,"vmag": 1.63},
    {"name": "Shaula","ra": 259.081,"dec": -37.100,"vmag": 1.63},
    {"name": "Bellatrix","ra": 81.334,"dec": 6.350,"vmag": 1.64},
    {"name": "Elnath","ra": 77.493,"dec": 28.618,"vmag": 1.65},
    {"name": "Miaplacidus","ra": 150.375,"dec": -69.658,"vmag": 1.69},
    {"name": "Alnilam","ra": 84.824,"dec": -1.201,"vmag": 1.70},
    {"name": "Alnair","ra": 328.055,"dec": -47.164,"vmag": 1.73},
    {"name": "Alioth","ra": 194.279,"dec": 55.949,"vmag": 1.77},
    {"name": "Suhail","ra": 139.387,"dec": -43.436,"vmag": 1.83},
    {"name": "Mirfak","ra": 49.620,"dec": 49.864,"vmag": 1.79},
    {"name": "Dubhe","ra": 165.717,"dec": 61.666,"vmag": 1.81},
    {"name": "Wezen","ra": 109.916,"dec": -26.981,"vmag": 1.83},
    {"name": "Kaus Australis","ra": 273.708,"dec": -34.230,"vmag": 1.85},
    {"name": "Avior","ra": 128.718,"dec": -59.336,"vmag": 1.86},
    {"name": "Alkaid","ra": 206.845,"dec": 49.314,"vmag": 1.86},
    {"name": "Atria","ra": 249.444,"dec": -69.043,"vmag": 1.90},
    {"name": "Alhena","ra": 100.089,"dec": 16.398,"vmag": 1.93},
    {"name": "Peacock","ra": 305.808,"dec": -70.473,"vmag": 1.94},
    {"name": "Alnitak","ra": 85.088,"dec": -1.943,"vmag": 1.98},
    {"name": "Alphard","ra": 141.670,"dec": -8.647,"vmag": 1.99},
    {"name": "Diphda","ra": 6.844,"dec": -17.994,"vmag": 2.04},
    {"name": "Kochab","ra": 229.412,"dec": 74.156,"vmag": 2.07},
    {"name": "Saiph","ra": 83.743,"dec": -9.670,"vmag": 2.07},
    {"name": "Nunki","ra": 286.046,"dec": -26.549,"vmag": 2.06},
    {"name": "Alpheratz","ra": 2.809,"dec": 29.090,"vmag": 2.06},
    {"name": "Ras Alhague","ra": 259.985,"dec": 12.585,"vmag": 2.08},
    {"name": "Shedar","ra": 9.493,"dec": 58.683,"vmag": 2.24},
    {"name": "Alderamin","ra": 323.492,"dec": 62.404,"vmag": 2.45},
    {"name": "Pherkard","ra": 236.425,"dec": 82.029,"vmag": 2.23},
    {"name": "Megrez","ra": 182.261,"dec": 56.382,"vmag": 3.32},
    {"name": "Eltanin","ra": 269.196,"dec": 51.493,"vmag": 2.24},
    {"name": "Polaris","ra": 37.954,"dec": 89.264,"vmag": 1.98},
    {"name": "Nusakan","ra": 264.819,"dec": 14.869,"vmag": 2.37},
    {"name": "Enif","ra": 328.675,"dec": 9.957,"vmag": 2.38},
    {"name": "Avior","ra": 128.718,"dec": -59.336,"vmag": 1.86},
    {"name": "Mizar","ra": 201.762,"dec": 54.929,"vmag": 2.04},
    {"name": "Alphecca","ra": 229.412,"dec": 26.702,"vmag": 2.22}
]


# (이전의 getstar, magnitude, _timeout_wrapper, run_with_timeout,
#  euclidean_distance, get_relative_pattern, get_catalog_pattern,
#  match_stars 함수는 그대로 유지)
def getstar(imagename):
    img = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{imagename} not found")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    blurred = cv2.GaussianBlur(enhanced, (3,3),0)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (1 < area < 500): continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * math.pi * (area / (perimeter*perimeter))
        if circularity < 0.2: continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r > 15: continue
        cx, cy = int(x), int(y)
        stars.append((cx, cy, round(r,2)))
    return stars

def magnitude(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"{image_path} not found")
    stars = getstar(image_path)
    star_data = []
    for i, (cx, cy, r) in enumerate(stars):
        background_annulus = np.zeros_like(gray_image)
        outer_radius = int(r) + 5
        cv2.circle(background_annulus, (cx, cy), outer_radius, 255, -1)
        star_area_mask = np.zeros_like(gray_image)
        cv2.circle(star_area_mask, (cx, cy), int(r), 255, -1)
        star_pixels = gray_image[star_area_mask > 0]
        background_pixels = gray_image[background_annulus > 0]
        local_background = np.mean(background_pixels) if background_pixels.size > 0 else 0
        total_star_brightness = np.sum(star_pixels) if star_pixels.size > 0 else 0
        pure_brightness = total_star_brightness - len(star_pixels) * local_background
        if pure_brightness > 0:
            mag_est = -2.5 * math.log10(pure_brightness) + 12.45
            star_data.append({
                'id': i+1,
                'magnitude': mag_est,
                'coords': (cx, cy),
                'radius': r
            })
    return star_data

def _timeout_wrapper(q, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        q.put(result)
    except Exception as e:
        q.put(e)

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    q = mp.Queue()
    p = mp.Process(target=_timeout_wrapper, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        return None
    return result

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_relative_pattern(stars):
    pattern = []
    for s1, s2 in combinations(stars, 2):
        d = euclidean_distance(s1['coords'], s2['coords'])
        pattern.append((s1['id'], s2['id'], d))
    return pattern

def get_catalog_pattern(nautical_stars):
    pattern = []
    for s1, s2 in combinations(nautical_stars, 2):
        d = math.sqrt((s1['ra'] - s2['ra'])**2 + (s1['dec'] - s2['dec'])**2)
        pattern.append((s1['name'], s2['name'], d))
    return pattern

def match_stars(observed_stars, nautical_stars, alpha=1.0, beta=1.0):
    results = []
    obs_pattern = get_relative_pattern(observed_stars)
    cat_pattern = get_catalog_pattern(nautical_stars)
    for obs in observed_stars:
        best_match = None
        min_score = float('inf')
        for cat in nautical_stars:
            mag_diff = abs(obs['magnitude'] - cat['vmag'])
            pattern_diff = 0
            for pair in obs_pattern:
                if obs['id'] in pair[:2]:
                    obs_dist = pair[2]
                    pattern_diff += min([abs(obs_dist - math.sqrt((cat['ra'] - c['ra'])**2 + (cat['dec'] - c['dec'])**2))
                                         for c in nautical_stars if c != cat])
            score = alpha*mag_diff + beta*pattern_diff
            if score < min_score:
                min_score = score
                best_match = cat
        results.append({
            'id': obs['id'],
            'ra': best_match['ra'],
            'dec': best_match['dec'],
            'magnitude': obs['magnitude'],
            'coords': obs['coords']
        })
    return results

class StarToGPS(torch.nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def cluster_positions(pred_positions, radius=0.1):
    pred_positions = np.array(pred_positions)
    if len(pred_positions) == 0:
        return []
    clusters = []
    used = np.zeros(len(pred_positions), dtype=bool)
    for i, pos in enumerate(pred_positions):
        if used[i]: continue
        dists = np.linalg.norm(pred_positions - pos, axis=1)
        cluster_idx = np.where(dists <= radius)[0]
        used[cluster_idx] = True
        cluster_center = np.mean(pred_positions[cluster_idx], axis=0)
        clusters.append(cluster_center)
    return clusters
    
if __name__ == "__main__":
    mpu6050_init()

    # 🔹 모델 준비 (입력 차원을 11에서 10으로 수정)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarToGPS(input_dim=10).to(device)
    model.load_state_dict(torch.load("2.pth", map_location=device))
    model.eval()

    # 🔹 센서 데이터 및 시간 정보 획득
    pitch_val, roll_val = get_pitch_roll()
    current_time = time.time()
    frac_day = (current_time % 86400) / 86400
    sin_t = np.sin(2 * np.pi * frac_day)
    cos_t = np.cos(2 * np.pi * frac_day)

    # 🔹 카메라로 사진 한 장 촬영 (libcamera-jpeg 이용)
    img_path = "capture.jpg"
    cmd = f"libcamera-jpeg -o {img_path} --width 640 --height 480 --camera 0"
    subprocess.run(shlex.split(cmd))

    # 🔹 촬영된 이미지 불러오기
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("❌ 사진 촬영 실패")
        exit()

    # 🔹 별 검출
    observed_stars = run_with_timeout(magnitude, args=(img_path,), timeout=5)
    if not observed_stars:
        print("❌ 별 검출 실패")
        exit()

    # 🔹 별 매칭
    matched_stars = match_stars(observed_stars, nautical_stars)
    h, w = frame.shape
    pred_positions = []

    for star in matched_stars:
        x_norm = star["coords"][0] / w
        y_norm = star["coords"][1] / h
        ra, dec = star["ra"], star["dec"]

        # 센서 데이터 및 시간 정보를 features 배열에 추가
        # Azimuth를 제외하고 pitch, roll, sin_t, cos_t를 포함
        features = np.array(
            [x_norm, y_norm, w, h, ra, dec, pitch_val, roll_val, sin_t, cos_t],
            dtype=np.float32
        )
        
        x_tensor = torch.tensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()[0]
        pred_positions.append(pred)

    if pred_positions:
        cluster_centers = cluster_positions(pred_positions)
        print("🌍 예측된 GPS 후보:", cluster_centers)
    else:
        print("❌ 추론된 별 없음")
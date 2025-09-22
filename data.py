# make_dataset.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from itertools import combinations, permutations
import multiprocessing as mp

# -----------------------------
# 항해력 기준별 예시 (실제 57개 사용 가능)
# -----------------------------
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

# ====================================
# getstar 함수 (이미지에서 별 추출)
# ====================================
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
        if not (1 < area < 500):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter*perimeter))
        if circularity < 0.2:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r > 15:
            continue
        cx, cy = int(x), int(y)
        stars.append((cx, cy, round(r,2)))
    return stars

# ====================================
# magnitude 함수 (겉보기 등급 추정)
# ====================================
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
            mag_est = -2.5 * math.log10(pure_brightness) + 8.45
            star_data.append({
                'id': i+1,
                'magnitude': mag_est,
                'coords': (cx, cy),
                'radius': r
            })
    return star_data

# ====================================
# 좌표 거리 계산
# ====================================
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ====================================
# 사진 별 간 상대 패턴 생성
# ====================================
from itertools import combinations

def get_relative_pattern(stars):
    pattern = []
    for s1, s2 in combinations(stars, 2):
        d = euclidean_distance(s1['coords'], s2['coords'])
        pattern.append((s1['id'], s2['id'], d))
    return pattern

# ====================================
# 항해력 별 간 상대 각도 패턴
# ====================================
def get_catalog_pattern(nautical_stars):
    pattern = []
    for s1, s2 in combinations(nautical_stars, 2):
        d = math.sqrt((s1['ra'] - s2['ra'])**2 + (s1['dec'] - s2['dec'])**2)
        pattern.append((s1['name'], s2['name'], d))
    return pattern

# ====================================
# 매칭 함수 (밝기 + 패턴)
# ====================================
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
            'coords': obs['coords'],
            'magnitude': obs['magnitude'],
            'match': best_match  # 전체 dict 포함
        })
    return results



def _timeout_wrapper(q, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        q.put(result)
    except Exception as e:
        q.put(e)

def run_with_timeout(func, args=(), kwargs={}, timeout=4):
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


class StarGPSDataset(Dataset):
    def __init__(self, image_folder, nautical_stars):
        self.samples = []
        files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg",".png"))]
        total_files = len(files)
        print(f"총 {total_files}개 파일 발견")

        for idx, fname in enumerate(files, 1):
            image_path = os.path.join(image_folder, fname)
            json_path = os.path.splitext(image_path)[0] + ".json"
            print(f"[{idx}/{total_files}] 처리 중: {fname}")

            if not os.path.exists(json_path):
                print(" ⚠️ JSON 없음, 건너뜀")
                continue

            with open(json_path, "r") as f:
                meta = json.load(f)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(" ⚠️ 이미지 로드 실패, 건너뜀")
                continue
            h, w = img.shape

            observed_stars = run_with_timeout(magnitude, args=(image_path,), timeout=4)
            if observed_stars is None:
                print(" ⏱ Timeout 발생, 건너뜀")
                continue

            if not observed_stars:
                print(" ⚠️ 밝은 별 없음, 건너뜀")
                continue

            # 매칭 수행
            matched_stars = match_stars(observed_stars, nautical_stars)

            for star, match in zip(observed_stars, matched_stars):
                x_norm = star['coords'][0] / w
                y_norm = star['coords'][1] / h
                ra, dec = 0.0, 0.0
                if match['match'] is not None:
                    # 매칭된 별의 RA/DEC 사용
                    cat_star = next((s for s in nautical_stars if s['name'] == match['match']), None)
                    if cat_star is not None:
                        ra = cat_star['ra']
                        dec = cat_star['dec']

                az = meta["sensor"]["azimuth"]
                pitch = meta["sensor"]["pitch"]
                roll = meta["sensor"]["roll"]
                t = meta["timestamp"] / 1000.0
                seconds_in_day = 24 * 3600
                frac_day = (t % seconds_in_day) / seconds_in_day
                sin_t = np.sin(2 * np.pi * frac_day)
                cos_t = np.cos(2 * np.pi * frac_day)

                features = np.array([x_norm, y_norm, w, h, ra, dec, az, pitch, roll, sin_t, cos_t], dtype=np.float32)
                target = np.array([meta["gps"]["latitude"], meta["gps"]["longitude"]], dtype=np.float32)

                self.samples.append((torch.tensor(features), torch.tensor(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ====================================
# 실행부
# ====================================
if __name__ == "__main__":
    dataset = StarGPSDataset("a", nautical_stars)
    print("샘플 개수:", len(dataset))
    torch.save(dataset.samples, "star_gps_dataset1.pt")
    print("Dataset saved to star_gps_dataset1.pt")

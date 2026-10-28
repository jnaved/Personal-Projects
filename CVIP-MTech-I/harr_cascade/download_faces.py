import urllib.request
import os
import cv2

urls = {
    "lincoln_portrait.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/ae/Abraham_Lincoln_head_on_shoulders_photo_portrait.jpg",
    "mona_lisa.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "einstein.jpg": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
    "curie.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/7e/Marie_Curie_c1920.jpg",
    "messi.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"
}

os.makedirs("harr_cascade/dataset", exist_ok=True)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for name, url in urls.items():
    dest = os.path.join("harr_cascade/dataset", name)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest, 'wb') as f:
            f.write(resp.read())
        img = cv2.imread(dest)
        if img is not None:
            # Resize if overly large to keep tests fast (max dimension 800)
            h, w = img.shape[:2]
            if max(h, w) > 800:
                scale = 800.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dest, img)
            print(f"Downloaded and standardized {name}: shape {img.shape}")
        else:
            print(f"Warning: {name} invalid image")
    except Exception as e:
        print(f"Failed {name}: {e}")

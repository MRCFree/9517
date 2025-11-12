import gc
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
import joblib

patches = []
K = 400
minibatch_size = 1000
BATCH_DESC = 40000
sift = cv2.SIFT_create(nfeatures=500)

class Patch:
    def __init__(self,path, c1, c2, label,iou, split):
        if c1 is None or c2 is None or label is None or iou is None or split is None:
            raise TypeError
        self.c1, self.c2, self.label, self.path, self.iou, self.split = c1, c2, label, path, float(iou), split
        self.img = None # will be generated after crop
        self.keypoints = None
        self.descriptor = None
        self.bow_histogram = None # can be None

    def pp(self):
        print(self.path, self.c1, self.c2, self.label, self.iou, self.split)

    # crop patch in image
    def crop_gray(self):
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        x1, y1 = self.c1
        x2, y2 = self.c2
        # prevent
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        self.img = img[y1:y2, x1:x2]
        return self.img

    def get_sift(self):
        if self.img is None:
            return None
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_img, None)
        self.keypoints, self.descriptor = kp, des
        return des

    def show_keypoints(self):
        if self.keypoints is None:
            self.get_sift()
        img_with_kp = cv2.drawKeypoints(
            self.img,  # 原图像
            self.keypoints,
            None,
            flags=0
        )

        cv2.imshow("SIFT keypoints", img_with_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show(self):
        if self.img is None:
            return
        cv2.imshow('patch',self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_patches(csvfile, num=None):
    out = []
    with open(csvfile, 'r', newline='') as f:
        lines = f.readlines()
        print(f"Start getting patches from {csvfile}...")
        for idx in range(1, min(num, len(lines)) if num is not None else len(lines)):
            line = lines[idx].split(',')
            # slice to coordinates
            c1 = (int(float(line[1])), int(float(line[2])))
            c2 = (int(float(line[3])), int(float(line[4])))
            path, label, iou, split = line[0], line[5], line[6], line[7]
            out.append(Patch(path, c1, c2, label, iou, split))

            print(f"patch{idx + 1} recieved, current patch: {idx + 1}/{len(lines)}. Patch is:{line[1:]}, pic location is:{line[0]}.")
        print(f"Finished! All patches from {csvfile} received")
    return out

def train_codebook_stream(patches, K=400):
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=minibatch_size, reassignment_ratio=0.01, verbose=1)
    buf = []
    total = 0
    for i, p in enumerate(patches):
        img = p.crop_gray()
        if img is None or img.size == 0:
            continue
        kp, des = sift.detectAndCompute(img, None)
        del img, kp
        if des is not None and len(des):
            buf.append(des)
            total += len(des)
        if sum(len(x) for x in buf) >= BATCH_DESC:
            kmeans.partial_fit(np.vstack(buf))
            buf.clear()
            gc.collect()
        del des
        if (i+1) % 2000 == 0:
            print(f'[partial_fit] processed {i+1} patches, descriptors so far ~{total}')
    if buf:
        kmeans.partial_fit(np.vstack(buf))
        buf.clear()
    joblib.dump(kmeans, 'kmeans_codebook.pkl')
    print('Codebook trained & saved.')
    return kmeans

def compute_bow(patches, kmeans):
    print('Computing BoW...')
    for i, p in enumerate(patches):
        img = p.crop()
        if img is None or img.size == 0:
            p.bow_histogram = None
            continue
        kp, des = sift.detectAndCompute(img, None)
        del img, kp
        if des is None or len(des) == 0:
            p.bow_histogram = np.zeros(kmeans.n_clusters, dtype=np.float32)
        else:
            idx = kmeans.predict(des)
            hist, _ = np.histogram(idx, bins=np.arange(kmeans.n_clusters+1), density=True)
            p.bow_histogram = hist.astype(np.float32)
        del des
        if (i+1) % 2000 == 0:
            print(f'[BoW] {i+1}/{len(patches)}')
        gc.collect()

def get_train_data(patches):
    X, Y = [], []
    for p in patches:
        if p.bow_histogram is not None:
            X.append(p.bow_histogram)
            Y.append(p.label)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int32)
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

def train_codebook_stream(patches, K=400, batch_desc=20000, checkpoint_every=5):
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=minibatch_size,
        reassignment_ratio=0.01,
        max_no_improvement=20,
        verbose=1
    )

    buf = np.empty((0, 128), dtype=np.float32)
    desc_used = 0
    pf_count = 0

    for i, p in enumerate(patches, 1):
        img = p.crop_gray()
        if img is None or img.size == 0:
            continue
        _, des = sift.detectAndCompute(img, None)
        del img  # 及时释放
        if des is None or len(des) == 0:
            continue

        # 追加到缓冲
        if buf.shape[0] == 0:
            buf = des.astype(np.float32, copy=False)
        else:
            buf = np.vstack((buf, des))

        desc_used += len(des)
        del des

        if buf.shape[0] >= batch_desc:
            kmeans.partial_fit(buf)
            pf_count += 1
            print(f'[partial_fit] used {desc_used} descriptors; batch {pf_count}')
            buf = np.empty((0, 128), dtype=np.float32)  # 重新分配空数组
            if pf_count % checkpoint_every == 0:
                joblib.dump(kmeans, f'kmeans_codebook_ckpt_{pf_count}.pkl')
            gc.collect()

    # 处理最后一批
    if buf.shape[0] > 0:
        kmeans.partial_fit(buf)
        pf_count += 1
        print(f'[partial_fit] final batch; total descriptors used {desc_used}')
        buf = np.empty((0, 128), dtype=np.float32)
        gc.collect()

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
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
import joblib

patches = []
sift = cv2.SIFT_create()
K = 400
minibatch_size = 1000

class Patch:
    def __init__(self,path, c1, c2, label,iou, split):
        self.c1, self.c2, self.label, self.path, self.iou, self.split = c1, c2, label, path, float(iou), split
        self.img = None # will be generated after crop
        self.keypoints = None
        self.descriptor = None
        self.bow_histogram = None

    def pp(self):
        print(self.path, self.c1, self.c2, self.label, self.iou, self.split)

    # crop patch in image
    def crop(self):
        img = cv2.imread(self.path)
        x1, y1 = self.c1
        x2, y2 = self.c2
        # prevent
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        self.img = img[y1:y2, x1:x2]

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

    def get_bow(self, visual_dictionary):
        if self.descriptor is None or len(self.descriptor) == 0:
            self.bow_histogram = None
            return
        word_ids = visual_dictionary.predict(self.descriptor)
        hist, _ = np.histogram(word_ids, bins=np.arange(K + 1))
        self.bow_histogram = hist
# 获取csv保存的patches

def get_patches(csvfile, num=None):
    with open(csvfile, 'r', newline='') as f:
        lines = f.readlines()
        for idx in range(1, min(num, len(lines)) if num is not None else len(lines)):
            line = lines[idx].split(',')
            # optimize below
            line[1] = (int(line[1]), int(line[2]))
            line[3] = (int(line[3]), int(line[4]))
            del line[2]
            del line[3]
            patches.append(Patch(*line))

def bow_extraction():
    valid_patches = []
    for patch in patches:
        patch.crop()  # 获取切片图像
        des = patch.get_sift()
        if des is not None:
            valid_patches.append(patch)
    descriptors = np.vstack([patch.descriptor for patch in valid_patches])
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=minibatch_size, random_state=None, verbose=1)
    kmeans.fit(descriptors)
    joblib.dump(kmeans, "kmeans_codebook.pkl")
    for patch in valid_patches:
        patch.get_bow(kmeans)

def get_train_data():
    train_X = [] # feature vector
    train_Y = [] # label
    for patch in patches:
        if patch.bow_histogram is None:
            continue
        train_X.append(patch.bow_histogram)
        train_Y.append(int(patch.label))
        X = np.array(train_X, dtype=np.float32)
        Y = np.array(train_Y, dtype=np.int32)
    return X, Y
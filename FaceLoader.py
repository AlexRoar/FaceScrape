import os
import urllib.request
import hashlib
import cv2
from skimage.transform import resize
from scipy.spatial import distance
import numpy as np
from copy import deepcopy
from shutil import copyfile
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FaceLoader:
    cascade_path = 'haarcascade_frontalface_alt2.xml'
    image_size = 160

    def __init__(self, url, margin=0.1, prefix='./', f_model=None):
        self.img_url = url
        self.margin = margin
        self.prefix = prefix
        self.url_hash = str(hashlib.md5(url.encode()).hexdigest())
        if not os.path.isdir(self.prefix + "tmp"):
            os.mkdir(self.prefix + "tmp")
        ext = url.split('.')[-1]
        self.local_url = self.prefix + "tmp/" + self.url_hash + "." + ext
        self.local_files = []
        try:
            self.downloadImg()
        except (KeyboardInterrupt, SystemExit):
            raise KeyboardInterrupt
        except:
            pass
        if f_model is None:
            f_model = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt',
                                               'models/res10_300x300_ssd_iter_140000.caffemodel')
            self.f_model = f_model
        else:
            self.f_model = deepcopy(f_model)

    def downloadImg(self):
        if not os.path.isdir(self.prefix + "tmp"):
            os.mkdir(self.prefix + "tmp")
        ext = self.img_url.split('.')[-1]
        self.local_url = self.prefix + "tmp/" + self.url_hash + "." + ext
        if "https://" in self.img_url or "http://" in self.img_url or "www." in self.img_url:
            urllib.request.urlretrieve(self.img_url, self.local_url)
        else:
            copyfile(self.img_url, self.local_url)
        self.local_files = [self.local_url]

    def __del__(self):
        for i in self.local_files:
            if os.path.isfile(i):
                os.remove(i)

    @staticmethod
    def prewhiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def load_and_align_image(self, margin=None, confidence=0.55):
        if margin is None:
            margin = self.margin
        aligned_images = []
        try:
            img = cv2.imread(self.local_url)
        except:
            try:
                self.downloadImg()
            except:
                return []
            img = cv2.imread(self.local_url)
        try:
            if img is None:
                return []
            if len(img.shape) != 3:
                return []
        except:
            return []

        resized, (top, left, ratio) = self.lossless_resize(img, 300)

        blob = cv2.dnn.blobFromImage(resized, 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.f_model.setInput(blob)
        detections = self.f_model.forward()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for face in detections[0, 0]:
            con = face[2]
            if con <= confidence:
                continue
            box = ((face[3:7] * 300 - np.array([left, top, left, top])) / ratio).astype("int")
            (startX_t, startY_t, endX_t, endY_t) = box.astype("int")
            (startX, startY, endX, endY) = (startX_t, startY_t, endX_t, endY_t)
            if endX < startX or endY < startY:
                continue

            h_rect = abs(endY - startY)
            w_rect = abs(endX - startX)
            m_w1, m_w2 = 0, 0
            m_h1, m_h2 = 0, 0
            if h_rect >= w_rect:
                m_w1, m_w2 = (h_rect - w_rect) // 2, (h_rect - w_rect) - (h_rect - w_rect) // 2
            else:
                m_h1, m_h2 = (w_rect - h_rect) // 2, (w_rect - h_rect) - (w_rect - h_rect) // 2
            startX = max(startX - m_w1 - int(margin * max(h_rect, w_rect)), 0)
            endX = min(endX + m_w2 + int(margin * max(h_rect, w_rect)), img.shape[1])
            startY = max(startY - m_h1 - int(margin * max(h_rect, w_rect)), 0)
            endY = min(endY + m_h2 + int(margin * max(h_rect, w_rect)), img.shape[0])

            try:
                cropped = img[startY: endY, startX: endX, :]
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    cropped = img[startY_t: endY_t, startX_t: endX_t, :]
            except:
                print("Failed face on", self.local_url)
                continue
            try:
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    print('Empty face:', self.img_url, (startX, startY, endX, endY))
                    continue
            except:
                continue
            aligned, _ = self.lossless_resize(cropped, self.image_size)
            aligned = resize(aligned, (self.image_size, self.image_size), mode='reflect')

            aligned_images.append(aligned)

        return np.array(aligned_images)

    def calc_embs(self, model, images=None, margin=None, batch_size=512):
        if margin is None:
            margin = self.margin
        if images is None:
            images = self.load_and_align_image(margin)
        if len(images) == 0:
            return []
        aligned_images = FaceLoader.prewhiten(images)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
        embs = FaceLoader.l2_normalize(np.concatenate(pd))
        return embs

    @staticmethod
    def calc_embs_static(model, images, batch_size=10):
        if len(images) == 0:
            return []
        aligned_images = FaceLoader.prewhiten(images)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
        embs = FaceLoader.l2_normalize(np.concatenate(pd))
        return embs

    @staticmethod
    def calc_dist(img_emb0, img_emb1):
        return distance.euclidean(img_emb0, img_emb1)

    @staticmethod
    def calc_dist_cosine(img_emb0, img_emb1):
        return distance.cosine(img_emb0, img_emb1)

    def lossless_resize(self, im, desired_size=300):
        old_size = im.shape[:2]

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                  value=color), (top, left, ratio)

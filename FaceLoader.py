import os
import urllib.request
import hashlib
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
import numpy as np


class FaceLoader:
    cascade_path = 'haarcascade_frontalface_alt2.xml'
    image_size = 160

    def __init__(self, url, margin=20, prefix='./'):
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

    def downloadImg(self):
        if not os.path.isdir(self.prefix + "tmp"):
            os.mkdir(self.prefix + "tmp")
        ext = self.img_url.split('.')[-1]
        self.local_url = self.prefix + "tmp/" + self.url_hash + "." + ext
        urllib.request.urlretrieve(self.img_url, self.local_url)
        self.local_files = [self.local_url]

    def __del__(self):
        for i in self.local_files:
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

    def load_and_align_image(self, margin=None):
        if margin is None:
            margin = self.margin
        cascade = cv2.CascadeClassifier(self.cascade_path)
        aligned_images = []
        try:
            img = imread(self.local_url)
        except (KeyboardInterrupt, SystemExit):
            raise KeyboardInterrupt
        except:
            try:
                self.downloadImg()
            except (KeyboardInterrupt, SystemExit):
                raise KeyboardInterrupt
            except:
                return []
            img = imread(self.local_url)
        if len(img.shape) != 3:
            return []
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except (KeyboardInterrupt, SystemExit):
            raise KeyboardInterrupt
        except:
            return []

        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for x, y, w, h in faces:
            try:
                cropped = img[max(y - margin // 2, 0): y + h + margin // 2,
                          max(x - margin // 2, 0): x + w + margin // 2, :]
            except (KeyboardInterrupt, SystemExit):
                raise KeyboardInterrupt
            except:
                continue
            aligned = resize(cropped, (self.image_size, self.image_size), mode='reflect')

            aligned_images.append(aligned)

        return np.array(aligned_images)

    def calc_embs(self, model, images=None, margin=None, batch_size=10):
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

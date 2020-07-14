import numpy as np
import tqdm
from FaceLoader import FaceLoader
import time
import json
import uuid


class SocialProcessor:
    def __init__(self, connection, model, batch=512):
        self.batch = batch
        self.connection = connection
        self.model = model
        self.size = FaceLoader.image_size

    def addRecords(self, data):
        for i, row in enumerate(data):
            data[i] = row + [str(uuid.uuid4())]
        c = self.connection.cursor()
        c.executemany('INSERT INTO global_table VALUES (?, ?, ?, ?, ?, ?, ?)', data)
        self.connection.commit()

    def addRecord(self, img_url, embeddings, created_at, user_id, service, img_area):
        c = self.connection.cursor()
        c.executemany('INSERT INTO global_table VALUES (?, ?, ?, ?, ?, ?, ?)',
                      [[img_url, embeddings, created_at, user_id, service, img_area, str(uuid.uuid4())]])
        self.connection.commit()

    def getFacesFromLinks(self, photo_links):
        faces = np.zeros((0, self.size, self.size, 3))
        links = []
        for url in tqdm.tqdm(photo_links):
            face_obj = FaceLoader(url, margin=20)
            faces_tmp = face_obj.load_and_align_image(margin=20)
            if len(faces_tmp) == 0:
                continue
            faces = np.vstack((faces, faces_tmp))
            links += [url] * len(faces)
            del face_obj
        return faces, links

    def processVkUser(self, api, ow_id, min_quality=3, des_type='x'):
        photo_links = []
        try:
            init_get = api.photos.getAll(owner_id=ow_id, extended=1, count=200)
        except:
            return
        count = init_get['count']
        if count == 0:
            return
        offset = 0
        while count > 0:
            for i in init_get['items']:
                for j in range(-1, -1 - min_quality, -1):
                    if i['sizes'][j]['type'] == des_type or j == -min_quality:
                        photo_links.append(i['sizes'][j]["url"])
                        print(i['sizes'][j]['type'])
                        break
            count -= len(init_get['items'])
            offset += len(init_get['items'])
            init_get = api.photos.getAll(owner_id=ow_id, extended=1, count=200, offset=offset)
        faces, links = self.getFacesFromLinks(photo_links)

        embedded = FaceLoader.calc_embs_static(self.model, images=faces, batch_size=self.batch)
        data = list(zip(embedded, links, faces))

        for i, row in enumerate(data):
            tmp_row = list(map(str, (
                row[1],
                json.dumps(list(map(float, list(row[0])))),
                time.time(),
                ow_id,
                "vk",
                json.dumps(row[2].tolist())
            )))
            data[i] = tmp_row

        self.addRecords(data)

    def loadBase(self):
        c = self.connection.cursor()
        return c.execute('SELECT * FROM global_table WHERE 1')

    def loadBaseEmbsUrls(self):
        c = self.connection.cursor()
        return c.execute('SELECT img_url, embeddings FROM global_table WHERE 1')

    def loadBaseEmbsImg(self):
        c = self.connection.cursor()
        return c.execute('SELECT img_url, img_area FROM global_table WHERE 1')

import numpy as np
import tqdm
from FaceLoader import FaceLoader
import time
import json
import uuid
import random
import os
import imageio
import cv2
from skimage import img_as_ubyte

from keras_facenet import FaceNet

embedder = FaceNet()


class SocialProcessor:
    def __init__(self, connection, model, batch=128, limit=1200, prefix='./'):
        self.batch = batch
        self.connection = connection
        self.model = model
        self.prefix = prefix
        self.size = FaceLoader.image_size
        self.limit = limit
        if not os.path.isdir("fragments"):
            os.mkdir("fragments")

    def addRecords(self, data):
        for i, row in enumerate(data):
            data[i] = list(row) + [str(uuid.uuid4())]

        data1 = []
        data2 = []

        for i, row in enumerate(data):
            data1.append(row[0:5] + [row[-1]])
            data2.append([row[-1], row[5]])

        c = self.connection.cursor(buffered=True)
        c.executemany('INSERT INTO global_table VALUES (%s,%s,%s,%s,%s,%s)', data1)
        self.connection.commit()

        for hash, content in data2:
            imageio.imwrite(self.prefix + 'fragments/' + hash + '.png', (content * 255).astype(np.uint8))

        print("Added %s rows" % (len(data)))

    def addRecord(self, img_url, embeddings, created_at, user_id, service, img_area):
        c = self.connection.cursor(buffered=True)
        hash = str(uuid.uuid4())
        c.executemany('INSERT INTO global_table VALUES (%s,%s,%s,%s,%s,%s)',
                      [[img_url, embeddings, created_at, user_id, service, hash]])
        self.connection.commit()
        img_area = cv2.cvtColor(img_area, cv2.COLOR_BGR2RGB)
        cv2.imwrite('fragments/' + hash + '.png', img_area)
        print("Added 1 row")

    def getFacesFromLinks(self, photo_links):
        faces = np.zeros((0, self.size, self.size, 3))
        links = []
        for url in tqdm.tqdm_notebook(photo_links):
            face_obj = FaceLoader(url, prefix=self.prefix)
            faces_tmp = face_obj.load_and_align_image()
            if len(faces_tmp) == 0:
                continue
            faces = np.vstack((faces, faces_tmp))
            links += [url] * len(faces_tmp)
            del face_obj
        return faces, links

    def processVkUser(self, api, ow_id, min_quality=2, des_type='z'):
        photo_links = []
        try:
            init_get = api.photos.getAll(owner_id=ow_id, extended=1, count=200)
        except (KeyboardInterrupt, SystemExit):
            raise KeyboardInterrupt
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
                        break
            count -= len(init_get['items'])
            offset += len(init_get['items'])
            init_get = api.photos.getAll(owner_id=ow_id, extended=1, count=200, offset=offset)
        print("Procesing", ow_id)
        photo_links = list(set(photo_links))
        faces, links = self.getFacesFromLinks(photo_links[:self.limit])

        embedded = FaceLoader.calc_embs_static(self.model, images=faces, batch_size=self.batch)
        data = list(zip(embedded, links, faces))

        for i, row in enumerate(data):
            tmp_row = [
                str(row[1]),
                str(json.dumps(list(map(float, list(row[0]))))),
                str(time.time()),
                str(ow_id),
                "vk",
                row[2]
            ]
            data[i] = tmp_row
        print("%s:" % (ow_id))
        self.addRecords(data)

    def loadBase(self):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT * FROM global_table WHERE 1')
        data = c.fetchall()
        c.close()
        return data

    def loadBaseEmbsUrls(self):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT img_url, embeddings FROM global_table WHERE 1')
        data = c.fetchall()
        c.close()
        return data

    def findMatches(self, embedding, threshold=1.0, batch=200):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT COUNT(scrape_id) FROM global_table WHERE 1')
        total = list(c.fetchall())[0][0]
        results = []
        embedding = np.array(embedding)
        for i in range(0, total // batch + 1):
            offset = i * batch
            c = self.connection.cursor(buffered=True)
            c.execute(
                'SELECT img_url, embeddings, created_at, user_id, service, scrape_id FROM global_table WHERE 1 LIMIT %s OFFSET %s' % (
                    batch, offset))
            data = c.fetchall()
            c.close()

            embs = np.array(list(map(json.loads, list(np.array(data)[:, 1]))))
            distances = FaceLoader.calc_dist(embs, np.expand_dims(embedding, axis=0))
            probs = 1 - FaceLoader.calc_dist_cosine(embs, np.expand_dims(embedding, axis=0))

            data = np.array(data)
            data = np.hstack([data, distances, probs])
            results = results + list(data)

        results = list(sorted(results, key=lambda x: x[-2], reverse=False))
        return results

    def addTask(self, id, service):
        hash = str(uuid.uuid4())
        created_at = time.time()
        c = self.connection.cursor(buffered=True)
        c.execute('INSERT INTO `process_query`(`id`, `service`, `created_at`, `query_id`) VALUES (%s, %s, %s, %s)',
                  [id, service, created_at, hash])
        self.connection.commit()
        c.close()

    def getTask(self, service="vk"):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT * FROM process_query WHERE service=%s LIMIT 1', [service])
        tasks = list(c.fetchall())
        c.close()
        return tasks[0]

    def getAllTasks(self, service="vk"):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT * FROM process_query WHERE service=%s', [service])
        tasks = list(c.fetchall())
        return tasks

    def delTask(self, query_id):
        c = self.connection.cursor(buffered=True)
        c.execute('DELETE FROM process_query WHERE query_id=%s', [query_id])
        self.connection.commit()
        c.close()

    def alreadyParsed(self, service="vk"):
        c = self.connection.cursor(buffered=True)
        c.execute('SELECT DISTINCT user_id FROM global_table WHERE service=%s', [service])
        tasks = list(c.fetchall())
        c.close()
        return set([i[0] for i in tasks])

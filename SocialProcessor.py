import numpy as np
import tqdm
from FaceLoader import FaceLoader
import time
import json
import uuid
import random
import os
import imageio
from skimage import img_as_ubyte


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
            imageio.imwrite(self.prefix + 'fragments/'+hash+'.png', img_as_ubyte(content))

        print("Added %s rows" % (len(data)))

    def addRecord(self, img_url, embeddings, created_at, user_id, service, img_area):
        c = self.connection.cursor(buffered=True)
        hash = str(uuid.uuid4())
        c.executemany('INSERT INTO global_table VALUES (%s,%s,%s,%s,%s,%s)',
                      [[img_url, embeddings, created_at, user_id, service, hash]])
        self.connection.commit()
        imageio.imwrite('fragments/' + hash + '.png', img_as_ubyte(img_area))
        print("Added 1 row")

    def getFacesFromLinks(self, photo_links):
        faces = np.zeros((0, self.size, self.size, 3))
        links = []
        for url in tqdm.tqdm_notebook(photo_links):
            face_obj = FaceLoader(url, margin=20, prefix = self.prefix)
            faces_tmp = face_obj.load_and_align_image(margin=20)
            if len(faces_tmp) == 0:
                continue
            faces = np.vstack((faces, faces_tmp))
            links += [url] * len(faces_tmp)
            del face_obj
        return faces, links

    def processVkUser(self, api, ow_id, min_quality=4, des_type='x'):
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
            random.shuffle(init_get['items'])
            init_get['items'] = init_get['items']
            for i in init_get['items']:
                for j in range(-1, -1 - min_quality, -1):
                    if i['sizes'][j]['type'] == des_type or j == -min_quality:
                        photo_links.append(i['sizes'][j]["url"])
                        break
            count -= len(init_get['items'])
            offset += len(init_get['items'])
            init_get = api.photos.getAll(owner_id=ow_id, extended=1, count=200, offset=offset)
        faces, links = self.getFacesFromLinks(photo_links[:self.limit])

        embedded = FaceLoader.calc_embs_static(self.model, images=faces, batch_size=self.batch)
        data = list(zip(embedded, links, faces))

        for i, row in enumerate(data):
            tmp_row =[
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
        for i in range(0, total // batch + 1):
            offset = i * batch
            c = self.connection.cursor(buffered=True)
            c.execute(
                'SELECT img_url, embeddings, created_at, user_id, service, scrape_id FROM global_table WHERE 1 LIMIT %s OFFSET %s' % (
                    batch, offset))
            data = c.fetchall()
            c.close()
            for row in data:
                emb = np.array(json.loads(row[1]))
                dist = FaceLoader.calc_dist(emb, embedding)
                row = list(row) + [dist]
                if dist > threshold:
                    continue
                else:
                    results.append(row)
        results = list(sorted(results, key=lambda x: x[-1], reverse=False))
        return results

    def addTask(self, id, service):
        hash = str(uuid.uuid4())
        created_at = time.time()
        c = self.connection.cursor(buffered=True)
        c.execute('INSERT INTO `process_query`(`id`, `service`, `created_at`, `query_id`) VALUES (%s, %s, %s, %s)', [id, service, created_at, hash])
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

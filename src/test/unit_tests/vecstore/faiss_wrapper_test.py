import unittest
import numpy as np
import time

from vecstore.hfaiss import faiss

class TestFaissWrapperClass(unittest.TestCase):

    def test_init(self):
        config = {
            'index_name': 'default0',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)
        # check init complete
        self.assertEqual(faiss_obj.isInitiated(), True)

        # check index name reflected
        self.assertEqual(faiss_obj.getIndexName(), config["index_name"])

    def test_process_spawn(self):
        config = {
            'index_name': 'default1',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)
        # check thread is alive
        self.assertEqual(faiss_obj.process_thread.is_alive(), True)

    def test_disk_read_write(self):
        config = {
            'index_name': 'default2',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)

        # generate disk data
        np.random.seed(faiss_obj.pq_random_seed)
        train_data = np.random.random((faiss_obj.pq_row_count, faiss_obj.dim)).astype('float32')
        ids = np.array(list(range(faiss_obj.pq_row_count))).astype('float32')
        train_disk_data = np.column_stack((train_data, ids))
        faiss_obj.train_disk_data = train_disk_data

        # index some data
        self.assertEqual(faiss_obj.writeVectorsToIndex(train_disk_data[:, :-1].astype('float32'), train_disk_data[:, -1:].astype('int').flatten()), True)
        
        # save to disk
        self.assertEqual(faiss_obj.saveModelToDisk(faiss_obj.model_location, faiss_obj.f_index), True)
        # read from disk
        self.assertEqual(faiss_obj.loadModelFromDisk(faiss_obj.model_location), True)
        
        # check if index is trained after disk read
        self.assertEqual(faiss_obj.isTrained(), True)
        # check indexed item count matches after disk read
        self.assertEqual((faiss_obj.getTotalItems() > 0) and (faiss_obj.getTotalItems() % faiss_obj.pq_row_count is 0), True)
        # crosscheck disk data after disk read
        self.assertEqual(np.array_equal(faiss_obj.train_disk_data, train_disk_data), True)

    def test_train_faiss(self):
        config = {
            'index_name': 'default3',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)

        np.random.seed(faiss_obj.pq_random_seed)
        faiss_obj.train_data = np.random.random((faiss_obj.pq_row_count, faiss_obj.dim)).astype('float32')

        # check training data is correctly generated
        self.assertEqual(faiss_obj.train_data.shape, (faiss_obj.pq_row_count, faiss_obj.dim))
        # check training success
        self.assertEqual(faiss_obj.trainFaiss(), True)

    def test_retrain_faiss(self):
        config = {
            'index_name': 'default4',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)

        np.random.seed(faiss_obj.pq_random_seed)
        train_data = np.random.random((faiss_obj.pq_row_count, faiss_obj.dim)).astype('float32')
        ids = np.array(list(range(faiss_obj.pq_row_count))).astype('float32')
        faiss_obj.train_disk_data = np.column_stack((train_data, ids))

        # check training data is correctly generated
        self.assertEqual(faiss_obj.train_disk_data.shape, (faiss_obj.pq_row_count, faiss_obj.dim + 1))
        
        # check if training success
        self.assertEqual(faiss_obj.retrainFaiss(), True)

    def test_add_documents(self):
        count_ = 101

        config = {
            'index_name': 'default5',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)

        np.random.seed(faiss_obj.pq_random_seed)
        train_data = np.random.random((count_, faiss_obj.dim)).astype('float32')
        ids = np.array(list(range(count_))).astype('float32')

        documents = []

        class Vector:
            def __init__(self, e):
                self.e = e

        class Document:
            def __init__(self, _id, e):
                self._id = str(id_)
                self.vector = Vector(e)

        for id_ in ids:
            id_ = int(id_)
            objd = Document(id_, train_data[id_].tolist())
            documents.append(objd)

        # add documents to process queue
        faiss_obj.addVectors(documents)
        time.sleep(10)

        # check indexed count
        self.assertEqual((faiss_obj.getTotalItems() > 0) and (faiss_obj.getTotalItems() % count_ is 0), True)

    def test_resize_dimension(self):

        config = {
            'index_name': 'default6',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        faiss_obj = faiss.Faiss(config)
        
        np.random.seed(faiss_obj.pq_random_seed)
        matrix_data_tiny = np.random.random((faiss_obj.pq_row_count, 400)).astype('float32')
        matrix_data_exact = np.random.random((faiss_obj.pq_row_count, 500)).astype('float32')
        matrix_data_bigger = np.random.random((faiss_obj.pq_row_count, 600)).astype('float32')

        # resize and check data shape
        self.assertEqual(faiss_obj.resizeForDimension(matrix_data_tiny).shape, (faiss_obj.pq_row_count, faiss_obj.dim))
        self.assertEqual(faiss_obj.resizeForDimension(matrix_data_exact).shape, (faiss_obj.pq_row_count, faiss_obj.dim))
        self.assertEqual(faiss_obj.resizeForDimension(matrix_data_bigger).shape, (faiss_obj.pq_row_count, faiss_obj.dim))

    def test_get_nearest(self):

        config = {
            'index_name': 'default7',
            'nlist': 1,
            'nprobe': 1,
            'bytesPerVec': 8,
            'bytesPerSubVec': 8,
            'dim': 8
        }

        k = 1
        rows = 10

        faiss_obj = faiss.Faiss(config)

        # generate disk data
        np.random.seed(faiss_obj.pq_random_seed)
        train_data = np.random.random((rows, faiss_obj.dim)).astype('float32')
        ids = np.array(list(range(rows))).astype('float32')
        train_disk_data = np.column_stack((train_data, ids))
        faiss_obj.train_disk_data = train_disk_data

        # index some data
        self.assertEqual(faiss_obj.writeVectorsToIndex(train_disk_data[:, :-1].astype('float32'), train_disk_data[:, -1:].astype('int').flatten()), True)

        # perform kNN query
        qresult = faiss_obj.getNearest(train_disk_data[:, :-1].astype('float32'), k)
        self.assertEqual( ( qresult[0], len(qresult[1]), len(qresult[2]), len(qresult[1][0]), len(qresult[2][0]) ), (True, rows, rows, k, k) )
        


if __name__ == '__main__':
    unittest.main()
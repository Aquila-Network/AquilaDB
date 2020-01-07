import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
import faiss
import yaml

import os
import threading
import queue
import time


class Faiss:
    def __init__(self, config):
        """Initialize FAISS index"""

        # load configurations
        self.index_name = config["index_name"]
        self.nlist = config["nlist"]
        self.nprobe = config["nprobe"]
        self.bytesPerVec = config["bytesPerVec"]
        self.bytesPerSubVec = config["bytesPerSubVec"]
        self.dim = config["dim"]

        # predefained quantizer training data size
        self.pq_row_count = 10001
        self.pq_random_seed = 1234

        # to keep the thread & queue running
        self.process_flag = True
        self.q_maxsize = 10100
        self.process_thread = None
        self._lock = threading.Lock()
        self.process_timeout_sec = 5 # in seconds

        # this is to keep track of all vectors inserted for retraining FAISS
        self.train_disk_data = None
        # set training data location
        self.train_disk_location = '/data/model_td_' + self.index_name
        # skip disk_data operations
        self.skip_disk_data = False
        # train with disk data
        self.run_disk_data_training = False

        # configure quantizer index
        logging.debug('FAISS init IndexFlatL2 quantizer')
        self.f_quantizer = faiss.IndexFlatL2(self.dim)
        if self.f_quantizer:
            logging.debug('FAISS init IndexFlatL2 quantizer success')
        else:
            logging.error('FAISS init IndexFlatL2 quantizer failed')

        # configure vector index
        logging.debug('FAISS init IndexIVFPQ index')
        self.f_index = faiss.IndexIVFPQ(self.f_quantizer, self.dim, self.nlist, self.bytesPerVec, self.bytesPerSubVec)
        if self.f_index:
            logging.debug('FAISS init IndexIVFPQ index success')
        else:
            logging.error('FAISS init IndexIVFPQ index failed')

        # set model location
        self.model_location = '/data/model_hf_' + self.index_name

        # load model from disk
        self.modelLoaded = self.loadModelFromDisk(self.model_location)

        # update model initialization status
        self.is_initiated = self.modelLoaded

        # if model is not loaded from disk, train a new index
        if not self.is_initiated:
            logging.debug('Index is not loaded from disk. Reverted to random initialization')

            # create a random matrix
            # self.train_data = np.asarray(matrix).astype('float32') # ref. for training actual data from python array
            np.random.seed(self.pq_random_seed)
            self.train_data = np.random.random((self.pq_row_count, self.dim)).astype('float32')

            # train index
            logging.debug('Index training started')
            training_status = self.trainFaiss()
            # update model initialization status
            self.is_initiated = training_status
            if training_status:
                logging.debug('Index training complete')
            else:
                logging.error('Index training failed')

        # spawn process thread
        logging.debug('Spawn FAISS process thread')
        self.spawn()


    def __del__(self):
        """Destruct FAISS"""

        # stop thread looping
        self.process_flag = False

        # join thread
        if self.process_thread:
            self.process_thread.join()


    def spawn (self):
        """Spawn process thread to process data from document queue and add it to index"""

        # create pipeline to add documents
        self.pipeline = queue.Queue(maxsize=self.q_maxsize)
        # create process thread
        self.process_thread = threading.Thread(target=self.process, args=(), daemon=True)
        # start process thread
        self.process_thread.start()
        # return self.pipeline


    def trainFaiss(self):
        """Train FAISS index"""

        # Lock index read / write until it is built
        with self._lock:
            logging.debug('FAISS train index started')
            self.f_index.train(self.train_data)
            logging.debug('FAISS train index finished')

            # write index to disk
            logging.debug('FAISS index writing started')
            self.modelLoaded = self.saveModelToDisk(self.model_location, self.f_index)
            logging.debug('FAISS index writing finished')
        
        # update model initialization status
        self.is_initiated = self.modelLoaded

        return self.is_initiated


    def retrainFaiss(self):
        """Retrain FAISS index with collected real data"""

        ret = False

        # create training data
        self.train_data = self.train_disk_data[:, :-1].astype('float32')

        # train
        ret = self.trainFaiss()

        # write data
        ret = self.writeVectorsToIndex(self.train_data, self.train_disk_data[:, -1:].astype('int').flatten())

        return ret

    def isInitiated(self):
        """Return True if index is already initialized"""

        return self.is_initiated

    def isTrained(self):
        """Return True if index is already trained"""

        return self.f_index.is_trained

    def getTotalItems(self):
        """Return number of items in faiss index"""

        return self.f_index.ntotal


    def getIndexName(self):
        """Return index name"""

        return self.index_name


    def loadModelFromDisk(self, location):
        """Load FAISS index from disk backup"""

        ret = True

        try:
            # read index
            self.f_index = faiss.read_index(location)
            logging.debug('FAISS index disk loading success')
            ret = True
        except Exception as e: 
            logging.warning('FAISS index disk loading failed')
            logging.warning(e)
            ret = False

        if not self.skip_disk_data:
            try:
                # read data
                self.train_disk_data = np.load(self.train_disk_location + '.npy')
                logging.debug('FAISS training data disk loading success')
                ret = True
            except Exception as e: 
                logging.warning('FAISS training data disk loading failed')
                logging.warning(e)
                ret = False
        
        return ret


    def saveModelToDisk(self, location, index):
        """Create a backup of FAISS index"""

        ret = True

        try:
            # write index
            faiss.write_index(index, location)
            logging.debug('FAISS index disk writing success')
            ret = True
        except Exception as e:
            logging.error('FAISS index disk writing failed')
            logging.error(e)
            ret = False
        
        if (not self.skip_disk_data) and (self.train_disk_data is not None):
            try:
                # write data
                np.save(self.train_disk_location, self.train_disk_data)
                logging.debug('FAISS training data disk writing success')
                ret = True
            except Exception as e:
                logging.error('FAISS training data disk writing failed')
                logging.error(e)
                ret = False

        return ret


    def addVectors(self, documents):
        """Add vectors to FAISS index"""

        ids = []

        logging.debug('Adding ' + str(len(documents)) + ' documents to process queue')
        # add vectors
        for document in documents:
            # add documents to the queue (Asynchronous)
            self.pipeline.put_nowait(document)
            ids.append(document._id)

        return True, ids


    def process(self):
        """The process thread"""

        while (self.process_flag):
            # print(list(self.pipeline.queue))

            # set a timeout till next vector indexing
            time.sleep(self.process_timeout_sec)

            # check if queue is not empty
            if self.pipeline.qsize() > 0:
                ids = []
                vecs = []

                # fetch all currently available documents from queue
                while not self.pipeline.empty():
                    # extract document & contents
                    document = self.pipeline.get_nowait()
                    _id = document._id
                    vec = document.vector
                    ids.append(_id)
                    vecs.append(vec.e)

                # convert to np matrix
                vec_data = np.asarray(vecs).astype('float32')
                id_data = np.asarray(ids).astype('int')
                # resize input matrix according to vector dimension config.
                vec_data = self.resizeForDimension(vec_data)

                # keep a copy for disk storage
                disk_list = np.column_stack((vec_data, id_data)).astype('float32')

                if not self.skip_disk_data:
                    # append to disk proxy
                    if self.train_disk_data is None:
                        self.train_disk_data = disk_list
                    else:
                        rows_before_append = self.train_disk_data.shape[0]
                        self.train_disk_data = np.append(self.train_disk_data, disk_list, axis=0)
                        # check if disk_data has enough data for retraining FAISS
                        if self.train_disk_data.shape[0] > self.pq_row_count:
                            # skip all disk_data activities from now on
                            self.skip_disk_data = True
                            if rows_before_append <= self.pq_row_count:
                                # run disk data training
                                self.retrainFaiss()
                
                # write data to index
                self.writeVectorsToIndex(vec_data, id_data)


    def writeVectorsToIndex(self, vec_data, id_data):
        """Write vectors to FAISS index"""

        ret = False

        # Lock index read / wtite until it is built
        with self._lock:
            # add vector
            self.f_index.add_with_ids(vec_data, id_data)
        
            # write to disk
            ret = self.saveModelToDisk(self.model_location, self.f_index)

        return ret


    def deleteVectors(self, ids):
        """Delete vectors from FAISS index"""

        return True, ids


    def getNearest(self, matrix, k):
        """Perform kNN search on FAISS index"""

        # convert to np matrix
        vec_data = np.asarray(matrix).astype('float32')
        # resize input matrix according to vector dimension config.
        vec_data = self.resizeForDimension(vec_data)
        
        # Lock index read / wtite until nearest neighbor search
        with self._lock:
            D, I = self.f_index.search(vec_data, k)
        return True, I.tolist(), D.tolist()


    def resizeForDimension(self, matrix_in):
        dtype_ = matrix_in.dtype

        # how much padding is remaining
        rem = self.dim - matrix_in.shape[1]

        # return itself if no padding is required
        if rem is 0:
            return matrix_in.astype(dtype_)
        # padd zeroes if needed
        elif rem > 0:
            return np.pad(matrix_in, ( (0,0), (0,rem) ), 'constant').astype(dtype_)
        # truncate if needed
        else:
            return matrix_in[ : , : self.dim].astype(dtype_)

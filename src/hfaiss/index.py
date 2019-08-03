import numpy as np
import faiss

model_location = '/data/VDB/model'

class Faiss:
    def __init__(self):
        self.nlist = 1
        self.nprobe = 1
        self.bytesPerVec = 8
        self.bytesPerSubVec = 8
        self.dim = 300
        self.modelLoaded = self.loadModelFromDisk(model_location)

    def initFaiss(self, nlist, nprobe, bytesPerVec, bytesPerSubVec, dim, matrix):
        self.nlist = nlist
        self.nprobe = nprobe
        self.bytesPerVec = bytesPerVec
        self.bytesPerSubVec = bytesPerSubVec
        self.dim = dim

        self.train_data = np.matrix(matrix).astype('float32')
        print('init quantizer', self.train_data, self.train_data.shape)
        self.f_quantizer = faiss.IndexFlatL2(self.dim)
        print('init index')
        self.f_index = faiss.IndexIVFPQ(self.f_quantizer, self.dim, self.nlist, self.bytesPerVec, self.bytesPerSubVec)
        print('train index')
        self.f_index.train(self.train_data)
        print('train index finished')

        self.modelLoaded = self.saveModelToDisk(self, model_location, self.f_index)
        return self.modelLoaded

    def loadModelFromDisk(self, location):
        try:
            # read index
            self.f_index = read_index(location)
            print('index loading success')
            return True
        except: 
            print('index loading failed')
            return False

    def saveModelToDisk(self, location, index):
        try:
            # write index
            faiss.write_index(index, location)
            print('index writing success')
            return True
        except:
            print('index writing failed')
            return False

    def addVectors(self, documents):
        ids = []
        vecs = []
        for document in documents:
            _id = document._id
            vec = document.vector
            ids.append(_id)
            vector_e = vec.e
            vector_e_l = len(vector_e)
            # check if the vector length is below dimention limit
            # then pad vector with 0 by dimension
            if vector_e_l < self.dim:
                vector_e.extend([0]*(self.dim-vector_e_l))
            # make sure vector length doesn't exceed dimension limit
            vecs.append(vector_e[:self.dim])
        # convert to np matrix
        vec_data = np.matrix(vecs).astype('float32')
        id_data = np.array(ids).astype('int')
        # add vector
        self.f_index.add_with_ids(vec_data, id_data)
        return True, ids

    def deleteVectors(self, ids):

        return True, ids

    def getNearest(self, matrix, k):
        # convert to np matrix
        vec_data = np.matrix(matrix).astype('float32')
        D, I = self.f_index.search(vec_data, k)
        return True, I.tolist(), D.tolist()
import logging

import grpc
from concurrent import futures
import time
import json

import proto.faiss_pb2_grpc as proto_server
import proto.faiss_pb2 as proto

from vecstore.indexman import FaissIndexManager

class FaissServicer (proto_server.FaissServiceServicer):
    def initFaiss (self, request, context):
        """Initialize FAISS index manager. This will manage multiple FAISS 
        indexes if necessary"""

        # prepare gRPC response
        response = proto.initFaissResponse()

        # faiss index manager instance
        self.faiss_index_manager = FaissIndexManager()

        # populate response with index manager initialization status
        if self.faiss_index_manager:
            logging.debug('Index manager instance creation success')
            response.status = True
        else:
            logging.error('Index manager instance creation failed')
            response.status = False

        return response

    def addVectors (self, request, context):
        """Add vectors to FAISS via index manager. Index manager decides to 
        which index a document to be added"""

        # prepare gRPC response
        response = proto.addVecResponse()

        documents = request.documents
        
        # add vectors to FAISS
        if self.faiss_index_manager:
            ret = self.faiss_index_manager.addVectors(documents)
        else:
            ret = [False, []]

        # check if insertion is success
        if ret[0]:
            logging.debug('Document insertion success')
        else:
            logging.error('Document insertion failed')

        # populate response
        response.status = ret[0]
        response._id.extend(ret[1])

        return response

    def deleteVectors (self, request, context):
        """Delete vectors from FAISS via index manager. Index manager decides 
        from which index a document to be deleted"""

        # prepare gRPC response
        response = proto.deleteVecResponse()

        # get list of ids to be deleted
        ids = request._id

        # delete vectors from FAISS
        if self.faiss_index_manager:
            ret = self.faiss_index_manager.deleteVectors(ids)
        else:
            ret = [False, []]

        # check if deletion is success
        if ret[0]:
            logging.debug('Document deletion success')
        else:
            logging.error('Document deletion failed')
        
        # populate response
        response.status = ret[0]
        response._id.extend(ret[1])
        return response

    def getNearest (self, request, context):
        """Perform ANN search on already indexed vectors."""

        # prepare gRPC response
        response = proto.getNearestResponse()

        # get search parameters from request
        matrix_in = request.matrix
        k = request.k

        matrix = []
        # convert rpc matrix_in to python matrix
        for vector in matrix_in:
            matrix.append(vector.e)

        # perform kNN search
        if self.faiss_index_manager:
            ret = self.faiss_index_manager.getNearest(matrix, k)
        else:
            ret = [False, [], []]

        # check if ANN is success
        if ret[0]:
            logging.debug('kNN search success')
        else:
            logging.error('kNN search failed')
        
        # populate response
        response.status = ret[0]
        response.ids = json.dumps(ret[1])
        response.dist_matrix = json.dumps(ret[2])
        return response

# configure gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers = 1),  options=[
          ('grpc.max_send_message_length', 100 * 1024 * 1024),
          ('grpc.max_receive_message_length', 100 * 1024 * 1024) ])
proto_server.add_FaissServiceServicer_to_server (FaissServicer(), server)

# start server
logging.info('Starting vecdb server. Listening on port 50052.')
server.add_insecure_port('[::]:50052')
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)

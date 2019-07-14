# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import vecdb_pb2 as vecdb__pb2


class VecdbServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.addDocuments = channel.unary_unary(
        '/vecdb.VecdbService/addDocuments',
        request_serializer=vecdb__pb2.addDocRequest.SerializeToString,
        response_deserializer=vecdb__pb2.addDocResponse.FromString,
        )
    self.deleteDocuments = channel.unary_unary(
        '/vecdb.VecdbService/deleteDocuments',
        request_serializer=vecdb__pb2.deleteDocRequest.SerializeToString,
        response_deserializer=vecdb__pb2.deleteDocResponse.FromString,
        )
    self.addNode = channel.unary_unary(
        '/vecdb.VecdbService/addNode',
        request_serializer=vecdb__pb2.addNodeRequest.SerializeToString,
        response_deserializer=vecdb__pb2.addNodeResponse.FromString,
        )
    self.getNearest = channel.unary_unary(
        '/vecdb.VecdbService/getNearest',
        request_serializer=vecdb__pb2.getNearestRequest.SerializeToString,
        response_deserializer=vecdb__pb2.getNearestResponse.FromString,
        )


class VecdbServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def addDocuments(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def deleteDocuments(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def addNode(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def getNearest(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_VecdbServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'addDocuments': grpc.unary_unary_rpc_method_handler(
          servicer.addDocuments,
          request_deserializer=vecdb__pb2.addDocRequest.FromString,
          response_serializer=vecdb__pb2.addDocResponse.SerializeToString,
      ),
      'deleteDocuments': grpc.unary_unary_rpc_method_handler(
          servicer.deleteDocuments,
          request_deserializer=vecdb__pb2.deleteDocRequest.FromString,
          response_serializer=vecdb__pb2.deleteDocResponse.SerializeToString,
      ),
      'addNode': grpc.unary_unary_rpc_method_handler(
          servicer.addNode,
          request_deserializer=vecdb__pb2.addNodeRequest.FromString,
          response_serializer=vecdb__pb2.addNodeResponse.SerializeToString,
      ),
      'getNearest': grpc.unary_unary_rpc_method_handler(
          servicer.getNearest,
          request_deserializer=vecdb__pb2.getNearestRequest.FromString,
          response_serializer=vecdb__pb2.getNearestResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'vecdb.VecdbService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
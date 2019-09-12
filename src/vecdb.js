const grpc = require('grpc')
var protoLoader = require('@grpc/proto-loader')

yaml = require('js-yaml')
fs   = require('fs')

// global variable, accessible from anywhere
// vecdb configuration
global.__g__vDBConfig = yaml.safeLoad(fs.readFileSync('./DB_config.yml', 'utf8'))
// modify configs from ENVIRONMENT variables
global.__g__vDBConfig.faiss.init.vecount = process.env.MIN_DOCS2INDEX || global.__g__vDBConfig.docs.vecount
// FAISS config.
global.__g__vDBConfig.faiss.init.nlist = process.env.MAX_CELLS || global.__g__vDBConfig.faiss.init.nlist
global.__g__vDBConfig.faiss.init.nprobe = process.env.VISIT_CELLS || global.__g__vDBConfig.faiss.init.nprobe
global.__g__vDBConfig.faiss.init.bpv = process.env.BYTES_PER_VEC || global.__g__vDBConfig.faiss.init.bpv
global.__g__vDBConfig.faiss.init.bpsv = process.env.BYTES_PER_SUB_VEC || global.__g__vDBConfig.faiss.init.bpsv
global.__g__vDBConfig.faiss.init.vd = process.env.FIXED_VEC_DIMENSION || global.__g__vDBConfig.faiss.init.vd
// authentication
global.__g__vDBConfig.couchDB.user = process.env.DB_USER || global.__g__vDBConfig.couchDB.user
global.__g__vDBConfig.couchDB.password = process.env.DB_PASSWORD || global.__g__vDBConfig.couchDB.password

// pouchdb instance
global.__g__iPDB = require('pouchdb')
__g__iPDB.plugin(require('pouchdb-find'))
// Initialize pouchDB by connecting to local instance of couchDB with default databases
global.__g__PDBs = {
    documentsDB: new __g__iPDB(__g__vDBConfig.couchDB.host + '/' + __g__vDBConfig.couchDB.DBInstance + '_docsdb'), // to keep all documents stored (synced on user request)
    mapperDB: new __g__iPDB(__g__vDBConfig.couchDB.host + '/' + __g__vDBConfig.couchDB.DBInstance + '_mapperdb'), // to keep vecID - docID mappings (not synced)
    swarmDB: new __g__iPDB(__g__vDBConfig.couchDB.host + '/' + __g__vDBConfig.couchDB.DBInstance + '_swarmdb'), // to keep swarm & peers data (not synced)
    replicationDB: new __g__iPDB(__g__vDBConfig.couchDB.host + '/' + __g__vDBConfig.couchDB.DBInstance + '_replicationdb'), // to keep replication data (not synced)
    sessionDB: new __g__iPDB(__g__vDBConfig.couchDB.host + '/' + __g__vDBConfig.couchDB.DBInstance + '_sessiondb') // to keep local session data to work properly (not synced)
}

// core utilities
const documentUtil = require('./core/document') // handle document operations
const searchUtil = require('./core/search') // handle vec/doc search operations
const swarmUtil = require('./core/swarm') // manage peers and swarm of vecDB
const eventUtil = require('./core/events') // handle events during document addition / replication
eventUtil.registerEvents() // register all events in the irst place

// setup grpc server for vecDB APIs
var PROTO_PATH = __dirname + '/proto/vecdb.proto'
var packageDefinition = protoLoader.loadSync(
    PROTO_PATH,
    {keepCase: true,
     longs: String,
     enums: String,
     defaults: true,
     oneofs: true
    })
 // Load corresponding interfaces from proto file
const proto = grpc.loadPackageDefinition(packageDefinition)
// create server instance
const server = new grpc.Server()

// service definitions of vecdb rpc API
server.addService(proto.vecdb.VecdbService.service, {
    // add new / modify existing docs into local DB and wait for eventual consistancy
    addDocuments(call, callback) {
        var documents = call.request.documents

        documentUtil.addOrModify(documents, (err, resp) => {
            if (!err) {
                callback (null, { status: true, _id: resp._id })
            }
            else {
                console.log('Error while adding document.')
                callback (null, { status: false, _id: null })
            }
        })
    },
    // delete existing docs from local DB and wait for eventual consistancy
    deleteDocuments(call, callback) {
        var ids = call.request._id
        
        documentUtil.delete(ids, (err, resp) => {
            if (!err) {
                callback (null, { status: true, _id: resp._id })
            }
            else {
                console.log('Error while deleting document.')
                callback (null, { status: false, _id: null })
            }
        })
    },
    // add this vecDB node to swarm
    addNode(call, callback) {
        var peers = call.request.peers
        
        swarmUtil.connectNewNodeToPeers(peers, (err, resp) => {
            if (!err) {
                callback (null, { status: true, peers: resp.peers})
            }
            else {
                console.log('Error while adding node to swarm.')
                callback (null, { status: false, peers: null })
            }
        })
    },
    // run KNN on local vedDB
    getNearest(call, callback) {
        var k = call.request.k
        var vector = call.request.matrix
        
        searchUtil.getKNDocumentsToVec(k, vector, (err, resp) => {
            if (!err) {
                callback (null, { status: true, matrix: resp.dist_matrix, documents: resp.documents })
            }
            else {
                console.log('Error while getting KNND.')
                callback (null, { status: false, matrix: null, documents: null })
            }
        })
    }
})

// server bind to a port and start it
server.bind('0.0.0.0:50051', grpc.ServerCredentials.createInsecure())
server.start()

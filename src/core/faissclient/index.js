const grpc = require('grpc')
const protoLoader = require('@grpc/proto-loader')
const njs = require('numjs')
const crypto = require('crypto')
const utils = require('../../utils')

// minimum documents required to train FAISS index
const FAISS_MIN_LIMIT = 10000

var PROTO_PATH = __dirname + '/../../proto/faiss.proto'
var packageDefinition = protoLoader.loadSync(
    PROTO_PATH,
    {keepCase: true,
     longs: String,
     enums: String,
     defaults: true,
     oneofs: true
    })

const proto = grpc.loadPackageDefinition(packageDefinition)

var faissRPC = new proto.faiss.FaissService('localhost:50052',
                grpc.credentials.createInsecure())

// temporarily keep vectors in memory until next faiss write
var faissTempVecStore = []

// global variables to keep track of FAISS index build
var faissIndexBuilt = false
var faissIndexBuildProgress = false

// get faiss status already
var sdb = __g__PDBs.sessionDB
sdb.get('local_faissStatus', (err, doc) => {
    if (doc) {
        faissIndexBuilt = doc.index_trained
    }
})

// temporarily keep vectors in memory until next faiss write
var annoyTempVecStore = []

// global variables to keep track of Annoy index build
var annoyIndexBuilt = false
var annoyIndexBuildProgress = false

// get annoy status already
var sdb = __g__PDBs.sessionDB
sdb.get('local_annoyStatus', (err, doc) => {
    if (doc) {
        annoyIndexBuilt = doc.index_trained
    }
})

function proto_matrix_transform (matrix_in) {
    var ret = []

    for (let r=0; r<matrix_in.length; r++) {
        var raw = matrix_in[r]
        ret.push({e: raw})
    }
    return ret
}

function initFaiss(nlist, nprobe, bpv, bpsv, vd, cbk) {
    var db = __g__PDBs.documentsDB
    utils.bulkGetVectors(db, (err, train_matrix) => {
        console.log(train_matrix.length, ' documents retrieved for faiss index training')
        if(!err) {
            var faiss_init_data = {
                "nlist": nlist,
                "nprobe": nprobe,
                "bpv": bpv,
                "bpsv": bpsv,
                "vd": vd,
                "matrix": proto_matrix_transform (train_matrix)
            }
            faissRPC.initFaiss(faiss_init_data, cbk)
        }
        else {
            cbk (err, null)
        }
    })
}

function addToFaiss_(matrix_in, cbk) {
    console.log(matrix_in.length)
    var err_ = false
    var counter = 0
    for (let i=0;i<matrix_in.length;i++) {
        var vector_in = matrix_in[i].m
        var faiss_add_data = {
            documents: [{
                _id: matrix_in[i].i,
                vector: {
                    e: vector_in
                }
            }]
        }
        faissRPC.addVectors(faiss_add_data, (err, resp) => {
            counter ++
            if (err) {
                err_ = err
            }
            if (counter === matrix_in.length) {
                cbk(err_, null)
            }
        })
    }
}

function mapVecDocId (doc_id, vec_id, cbk) {
    // get & increment document count locally in DB
    var mdb = __g__PDBs.mapperDB
    utils.mapDoc2VecID(mdb, doc_id, vec_id, (err) => {
        if (!err) {
            utils.mapVec2DocID(mdb, doc_id, vec_id, (err) => {
                if (!err) {
                    // success mapping done
                    cbk (null)
                }
                else {
                    // failed mapping
                    cbk (err)
                }
            })
        }
        else {
            cbk (err)
        }
    })
}

function getVecId (cbk) {
    // get & increment document count locally in DB
    utils.getVecID((err, counter_) => {
        if (!err) {
            cbk (null, counter_)
        }
        else {
            console.log(err)
            cbk(err, null)
        }
    })
}

function addToFaiss(new_matrix, vec_id, cbk) {
    var init_config = __g__vDBConfig.faiss.init
    // faiss index is not built yet
    if (!faissIndexBuilt && !faissIndexBuildProgress) {
        faissIndexBuildProgress = true
        // keep data for addition to faiss
        faissTempVecStore.push({m: new_matrix, i: vec_id})
        // train index only once and update db session
        initFaiss(init_config.nlist, init_config.nprobe, init_config.bpv, init_config.bpsv, init_config.vd, (err, resp) => {
            if (!err) {
                // update local status about faiss initialization
                sdb.put({_id:'local_faissStatus', index_trained: true}, (err, resp) => {
                    if (!err) {
                        faissIndexBuildProgress = false
                        faissIndexBuilt = true

                        // add pending data to faiss
                        var tmp_arr = faissTempVecStore
                        faissTempVecStore = []

                        addToFaiss_(tmp_arr, (err, resp) => {
                            if (!err) {
                                // console.log("added to faiss")
                            }
                            else {
                                console.log("can't add vector to faiss", err)
                            }
                        })
                    }
                })
            }
            else {
                console.log(err)
            }
        })
    }
    // faiss indexing is in progress
    if (!faissIndexBuilt && faissIndexBuildProgress) {
        faissTempVecStore.push({m: new_matrix, i: vec_id})
    }
    // faiss index is built. We are ready to add new docs
    if (faissIndexBuilt && !faissIndexBuildProgress) {
        faissTempVecStore.push({m: new_matrix, i: vec_id})

        // add pending data to faiss
        var tmp_arr = faissTempVecStore
        faissTempVecStore = []

        addToFaiss_(tmp_arr, (err, resp) => {
            if (!err) {
                // console.log("added to faiss")
            }
            else {
                console.log("can't add vector to faiss", err)
            }
        })
    }
}



function addToAnnoy(new_matrix, vec_id, cbk) {
    var init_config = __g__vDBConfig.faiss.init
    // annoy index is not built yet
    if (!annoyIndexBuilt && !annoyIndexBuildProgress) {
        annoyIndexBuildProgress = true
        // keep data for addition to annoy
        annoyTempVecStore.push({m: new_matrix, i: vec_id})
        // train index only once and update db session
        initFaiss(0, 0, 0, 0, init_config.vd, (err, resp) => {
            if (!err) {
                // update local status about annoy initialization
                sdb.put({_id:'local_annoyStatus', index_trained: true}, (err, resp) => {
                    if (!err) {
                        annoyIndexBuildProgress = false
                        annoyIndexBuilt = true

                        // add pending data to annoy
                        var tmp_arr = annoyTempVecStore
                        annoyTempVecStore = []

                        addToFaiss_(tmp_arr, (err, resp) => {
                            if (!err) {
                                // console.log("added to annoy")
                            }
                            else {
                                console.log("can't add vector to annoy", err)
                            }
                        })
                    }
                })
            }
            else {
                console.log(err)
            }
        })
    }
    // annoy indexing is in progress
    if (!annoyIndexBuilt && annoyIndexBuildProgress) {
        annoyTempVecStore.push({m: new_matrix, i: vec_id})
    }
    // annoy index is built. We are ready to add new docs
    if (annoyIndexBuilt && !annoyIndexBuildProgress) {
        annoyTempVecStore.push({m: new_matrix, i: vec_id})

        // add pending data to annoy
        var tmp_arr = annoyTempVecStore
        annoyTempVecStore = []

        addToFaiss_(tmp_arr, (err, resp) => {
            if (!err) {
                // console.log("added to annoy")
            }
            else {
                console.log("can't add vector to annoy", err)
            }
        })
    }
}

module.exports = {
    // add a new vector to faiss db
    addNewVector (doc_id, new_matrix) {
        // only add to faiss if document is fresh
        getVecId((err, vec_id) => {
            if (!err) {
                mapVecDocId (doc_id, vec_id, (err) => {
                    if(!err) {
                        var init_config = __g__vDBConfig.faiss.init
                        
                        // check if vector count reached trainable min. requirement
                        if (vec_id > FAISS_MIN_LIMIT) {
                            addToFaiss(new_matrix, vec_id, (err, resp) => {
                                if (!err) {
                                    console.log('Added vectors to FAISS')
                                }
                                else {
                                    console.log('Error adding vectors to FAISS', err, resp)
                                } 
                            })    
                        }
                        // check if vector count reached trainable min. requirement
                        else if (vec_id >= init_config.vecount) {
                            if (vec_id === init_config.vecount) {
                                // copy initial vectors collected so far for annoy initialization
                                annoyTempVecStore = faissTempVecStore
                            }

                            addToAnnoy(new_matrix, vec_id, (err, resp) => {
                                if (!err) {
                                    console.log('Added vectors to Annoy')
                                }
                                else {
                                    console.log('Error adding vectors to Annoy', err, resp)
                                } 
                            })  

                            // keep vector to be added in memory
                            faissTempVecStore.push({m: new_matrix, i: vec_id})  
                        }
                        else {
                            // keep vector to be added in memory
                            faissTempVecStore.push({m: new_matrix, i: vec_id})
                        }
                    }
                })
            }
            else {
                console.log("Can't get vec ID")
            }
        })
    },
    getKNN(k, matrix, cbk){
        var faiss_search_data = {
            "matrix": matrix,
            "k": k
        }
        
        faissRPC.getNearest(faiss_search_data, (err, resp) => {
            var db = __g__PDBs.documentsDB
            var mdb = __g__PDBs.mapperDB

            if (!err) {
                var dist_matrix_str = resp.dist_matrix
                var vec_ids = JSON.parse(resp.ids)
                var vec_ids_ = []
                
                // create ids list
                for(let i=0;i<vec_ids.length;i++){
                    for(let j=0;j<vec_ids[i].length;j++){
                        vec_ids_.push(''+vec_ids[i][j])
                    }
                }

                // get real doc ids from mapping
                mdb.allDocs({
                    include_docs: true,
                    keys: vec_ids_
                }, function(err, resp) {
                    if(!err) {
                        var doc_ids_ = []
                        for(let i=0;i<resp.rows.length;i++){
                            doc_ids_.push(resp.rows[i].doc.val)
                        }

                        // get real docs
                        db.allDocs({
                            include_docs: true,
                            keys: doc_ids_
                        }, function(err, resp) {
                            if(!err) {
                                resp_ = {
                                    status: true,
                                    dist_matrix: dist_matrix_str,
                                    documents: Buffer.from(JSON.stringify(resp.rows))
                                }
                                cbk(err, resp_)
                            }
                            else{
                                cbk(err, resp)
                            }
                        })
                    }
                    else{
                        cbk(err, resp)
                    }
                })
            }
            else {
                cbk (err, null)
            }
        })
    },
    // delete a vector from faiss db
    removeVector (doc_id) {
        console.log('To remove: ', doc_id)
        // map doc id to faiss id
        var mdb = __g__PDBs.mapperDB
        utils.getIDMap(mdb, doc_id, (doc) => {
            // remove vector
            var faissDeleteData = {
                _id: [doc.val]
            }
            faissRPC.deleteVectors(faissDeleteData, (err, resp) => {
                if (err) {
                    console.log('error deleting item from vector index', err)
                }
            })
        })  
    }
}

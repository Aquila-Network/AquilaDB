// TBD: currently vector id is kept inside a global variable and slowly updated to DB
// to eliminate the problem of conflict. This approach is not good in case of multiple 
// instances of vecDB runs behind a load balancer. Then we need some other mechanism like 
// an external service to sync. read writes.

// global variable to keep vector ID
var _vec_ID = null
// init worker
runVecIDWorker ()

function runVecIDWorker () {
    console.log('running VecID Worker')
    var sdb = __g__PDBs.sessionDB
    // check if doc already availavle
    sdb.get('local_vecID', function(err, doc) {
        var counter_ = null
        var data = {
            _id: 'local_vecID'
        }

        if (doc) {
            // if it is in initial stage
            if (_vec_ID === null) {
                if (doc.vector_count) {
                    _vec_ID = doc.vector_count
                }
                else {
                    _vec_ID = 0
                }
            }
            data['_rev'] = doc._rev
        }

        // set vecID to counter
        data['vector_count'] = _vec_ID
        
        sdb.put(data, function(err, response) {
            if (err) { return console.log(err) }
            // run recursively 
            setTimeout(function(){ runVecIDWorker () }, __g__vDBConfig.vectorID.sync_t)
        })
    })
}

module.exports = {
    documentUpdate: function (db, payload, cbk) {
        // check if doc already availavle
        db.get(payload._id, (err, doc) => {
            if(!err) {
                // already exists
                payload._rev = doc._rev
                // update latest revision
                db.put(payload, cbk)
            }
            else {
                // create new if not found error
                if (err.status === 404) {
                    db.put(payload, cbk)
                }
                else {
                    cbk(err, doc)
                }
            }
        })
    },
    getIDMap: function (mdb, id_in, cbk) {
        mdb.get(id_in, (err, doc) => {
            if(!err) {
                cbk(doc)
            }
            else {
                cbk(null)
            }
        })
    },
    getVecID: function (cbk) {
        // make sure id is in sync with DB
        if (_vec_ID !== null) {
            _vec_ID = _vec_ID + 1
            cbk (null, _vec_ID)
        }
        else {
            cbk ('ID is not synced with local DB', _vec_ID)
        }
    },
    mapDoc2VecID (mdb, doc_id, vec_id, cbk) {
        var data = {
            _id: ''+doc_id,
            val: vec_id
        }
        this.documentUpdate(mdb, data, function(err, doc) {
            if (err) {
                console.log('Doc2Vec mapping failed')
            }
            cbk(err)
        })
    },
    mapVec2DocID (mdb, doc_id, vec_id, cbk) {
        var data = {
            _id: ''+vec_id,
            val: doc_id
        }
        this.documentUpdate(mdb, data, function(err, doc) {
            if (err) {
                console.log('Vec2Doc mapping failed')
            }
            cbk(err)
        })
    },
    bulkGetVectors (db, cbk) {
        db.allDocs({
            // selector: {_id: {$gt: null}},
            // fields: ['vector']
            include_docs: true
          }, function (err, result) {
              console.log(err, result)
            // handle result
            f_array = []
            for(let i=0;i<result.rows.length;i++) {
                f_array.push(result.rows[i].doc.vector)
            }
            cbk (err, f_array)
        })
    }
}
const crypto = require('crypto')
const utils = require('../../utils')

module.exports = {
    // add new docs, modify if already present
    addOrModify (documents, cbk) {
        var doclength = documents.length
        var itrcnt = 0
        var id_list = []
        // get data dor each docs
        documents.forEach((document) => {
            // get vector
            var vector = {}
            if (document.vector && document.vector.e) {
                vector = document.vector.e

                // get document ID
                var _id = null
                if (document._id) {
                    _id = document._id
                }
                else {
                    // generate ID from vector
                    _id = crypto.createHash('md5').update(JSON.stringify(vector)).digest('hex')
                }

                // get document data
                var payload = {}
                if (document.b64data) {
                    try {
                        payload = JSON.parse(document.b64data.toString("utf-8"))
                    }
                    catch (err) {
                        console.log(err)
                        console.log('Adding empty metadata..')
                        payload = {}
                    }
                }
                // add more data to store in DB
                payload._id = _id
                payload.vector = vector

                var db = __g__PDBs.documentsDB
                utils.documentUpdate (db, payload, (err, response) => {
                    if (!err) {
                        // success update
                        id_list.push(_id)
                        // console.log('success update')
                        // count iteration
                        itrcnt ++
                        if (itrcnt === doclength) {
                            cbk (null, {_id:id_list})
                        }
                    }
                    else {
                        // failed update
                        console.log(err)
                        // count iteration
                        itrcnt ++
                        if (itrcnt === doclength) {
                            cbk (null, {_id:id_list})
                        }
                    }
                })
            }
            else {
                // no vector data
                itrcnt ++
                if (itrcnt === doclength) {
                    cbk (null, {_id:id_list})
                }
            }
        })
    },
    delete (documentIds, cbk) {
        var doclength = documentIds.length
        var itrcnt = 0
        var id_list = []
        documentIds.forEach((_id) => {
            // find and delete docs
            var db = __g__PDBs.documentsDB
            db.get(_id, (err, doc) => {
                if(!err) {
                    db.remove(doc, (err, response) => {
                        if (!err) {
                            // remove is success
                            id_list.push(_id)
                            console.log('remove is success')
                            itrcnt ++
                            if (itrcnt === doclength) {
                                cbk (null, {_id: id_list})
                            }
                        }
                        else {
                            // remove is unsuccess
                            console.log('remove is unsuccess', err)
                            itrcnt ++
                            if (itrcnt === doclength) {
                                cbk (null, {_id: id_list})
                            }
                        }
                    })
                }
                else {
                    // create new if not found error
                    if (err.status === 404) {
                        // nothing to remove
                        console.log('nothing to remove')
                        itrcnt ++
                        if (itrcnt === doclength) {
                            cbk (null, {_id: id_list})
                        }
                    }
                    else {
                        // failed to remove
                        console.log('failed to remove', err)
                        itrcnt ++
                        if (itrcnt === doclength) {
                            cbk (null, {_id: id_list})
                        }
                    }
                }
            })
        })
    }
}

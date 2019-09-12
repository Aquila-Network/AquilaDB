const faiss_client = require('../faissclient')

module.exports = {
    // handle change event
    // TBD: now it treats only one document is changed at a time.
    // if multiple documents are changed at a time, logic here should 
    // be chabged.
    handle(change) {
        // for(let i=0; i<change.changes.length; i++){
            // get the number of times the same document is updated.
            var rev = change.doc._rev
            var r_times = rev.split('-')[0] // change.changes[i].rev.split('-')[0]
            var id = change.doc._id
            var vector = change.doc.vector
            var revTimes = rev.split('-')[0]
            // check if the change is delete / modify
            if (!change.deleted) {
                // document is created / modified
                // check if document is created - check revision times is 1
                if (revTimes === '1') {
                    faiss_client.addNewVector(id, vector)
                }
                else {
                    console.log('Document with id: ' + id + ' is not fresh!')
                    // currently, an unfresh document is not sent to faiss. 
                    // But when networking enabled, you might want to send 
                    // it to faiss for indexing
                }
            }
            else {
                // document deleted, remove vector from faiss
                faiss_client.removeVector(id)
            }
        // }
    }
}
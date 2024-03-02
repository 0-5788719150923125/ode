import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({ backend: 'tensorflow' })
    } else {
        // browser
        const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        })
        worker.postMessage({ backend: 'webgl' })
    }
})()

import { trainModel } from './dev-train.js'
;(async function () {
    // if node.js
    if (typeof window === 'undefined') {
        await trainModel('tensorflow')
        // if browser
    } else {
        const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        })
        worker.postMessage({ command: 'train' })
    }
})()

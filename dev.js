import { trainModel } from './dev-engine.js'
;(async function () {
    if (typeof window === 'undefined') {
        // if node.js
        await trainModel()
    } else {
        // if browsers
        const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        })
        worker.postMessage({ command: 'train' })
    }
})()

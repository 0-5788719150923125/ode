import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            backend: 'tensorflow',
            batchSize: 8,
            gradientAccumulationSteps: 2,
            generateEvery: 64,
            sampleLen: 32
        })
    } else {
        // browser
        const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        })
        worker.postMessage({
            backend: 'webgl',
            batchSize: 1,
            gradientAccumulationSteps: 128,
            generateEvery: 0
        })
    }
})()

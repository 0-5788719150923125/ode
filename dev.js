import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            backend: 'tensorflow',
            batchSize: 64,
            gradientAccumulationSteps: 2,
            generateEvery: 32,
            sampleLen: 64,
            predictLength: 16
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

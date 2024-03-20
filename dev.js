import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            backend: 'tensorflow',
            batchSize: 1,
            gradientAccumulationSteps: 64,
            generateEvery: 512,
            sampleLen: 64
            // overfit: 100
            // loadFromFile: 'data/models/ode'
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

import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            backend: 'tensorflow',
            batchSize: 32,
            gradientAccumulationSteps: 4,
            generateEvery: 256,
            sampleLen: 96
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

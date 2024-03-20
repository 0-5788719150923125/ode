import { trainModel } from './dev-train.js'
;(async function () {
    // using node.js
    if (typeof window === 'undefined') {
        await trainModel({
            version: 4,
            backend: 'tensorflow',
            batchSize: 4,
            gradientAccumulationSteps: 256,
            generateEvery: 256,
            sampleLen: 128
            // overfit: 100
            // loadFromFile: 'data/models/ode'
        })
    }
    // using browser
    else {
        new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        }).postMessage({
            backend: 'cpu',
            batchSize: 1,
            gradientAccumulationSteps: 4,
            generateEvery: 4
        })
    }
})()

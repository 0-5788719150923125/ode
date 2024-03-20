import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            version: 3,
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

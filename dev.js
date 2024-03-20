import { trainModel } from './dev-train.js'
;(async function () {
    if (typeof window === 'undefined') {
        // node.js
        await trainModel({
            version: 4,
            backend: 'tensorflow',
            batchSize: 2,
            gradientAccumulationSteps: 512,
            generateEvery: 512,
            sampleLen: 256
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

import { trainModel } from './dev-train.js'
;(async function () {
    // using node.js
    if (typeof window === 'undefined') {
        await trainModel({
            version: 5,
            backend: 'tensorflow',
            batchSize: 2,
            gradientAccumulationSteps: 16,
            generateEvery: 128,
            sampleLength: 256
            // learningRate: 0.0022
        })
    }
    // using browser
    else {
        new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        }).postMessage({
            backend: 'webgl',
            version: 4,
            batchSize: 1,
            gradientAccumulationSteps: 512,
            generateEvery: 512,
            sampleLength: 256
        })
    }
})()

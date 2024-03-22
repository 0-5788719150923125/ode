import { trainModel } from './dev-train.js'
;(async function () {
    // using node.js
    if (typeof window === 'undefined') {
        await trainModel({
            version: 5,
            backend: 'tensorflow',
            batchSize: 1,
            gradientAccumulationSteps: 64,
            generateEvery: 256,
            sampleLen: 256
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

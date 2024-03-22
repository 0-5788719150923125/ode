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
            sampleLen: 256
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
            sampleLen: 256
        })
    }
})()

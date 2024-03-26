import { trainModel } from './dev-train.js'
;(async function () {
    const testArgs = {
        version: 5,
        batchSize: 2,
        gradientAccumulationSteps: 16,
        generateEvery: 128,
        sampleLength: 256
        // learningRate: 0.0022
    }
    // using node.js
    if (typeof window === 'undefined') {
        await trainModel({
            backend: 'tensorflow',
            ...testArgs
        })
    }
    // using browser
    else {
        new Worker(new URL('dev-worker.js', import.meta.url), {
            type: 'module'
        }).postMessage({
            backend: 'webgl',
            ...testArgs
        })
    }
})()

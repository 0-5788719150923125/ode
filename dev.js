import { trainModel } from './dev-train.js'

const testArgs = {
    version: 5,
    batchSize: 2,
    gradientAccumulationSteps: 16,
    generateEvery: 128,
    sampleLength: 256
}

;(async function () {
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

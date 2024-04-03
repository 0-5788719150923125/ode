import { trainModel } from './dev-train.js'

const testArgs = {
    version: 7,
    batchSize: 2,
    gradientAccumulationSteps: 32,
    generateEvery: 4,
    sampleLength: 256,
    predictLength: 64,
    debug: false
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

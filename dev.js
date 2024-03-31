import { trainModel } from './dev-train.js'

const testArgs = {
    version: 4,
    batchSize: 1,
    gradientAccumulationSteps: 16,
    generateEvery: 256,
    sampleLength: 512,
    predictLength: 128,
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

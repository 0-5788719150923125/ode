import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    const net = new ODE({
        backend: 'webgl',
        layout: [128, 128],
        learningRate: 1e-3,
        decay: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 64
    })
    await net.init()

    console.log(net.model.summary())

    const trainArgs = {
        batchSize: 1,
        gradientAccumulationSteps: 128,
        sampleLen: 64,
        generateEvery: 1024
    }

    const dataset = stringSampler(sampleLen)

    await net.train(dataset, trainArgs)
}

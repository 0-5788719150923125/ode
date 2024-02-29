import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    console.log(event.data)

    const net = new ODE({
        backend: 'webgl',
        layout: [128, 128, 128],
        learningRate: 1e-3,
        decayRate: 0.9,
        momentum: 0.01,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 16
    })
    await net.init()

    console.log(net.model.summary())

    const batchSize = 512
    const sampleLen = 64
    const dataset = stringSampler(sampleLen)

    await net.train(dataset, batchSize, sampleLen)
}

import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    console.log(event.data)

    const net = new ODE({
        backend: 'webgl',
        layout: [128, 128, 128],
        learningRate: 1e-5,
        decayRate: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 64
    })
    await net.init()

    console.log(net.model.summary())

    const sampleLen = 64
    const dataset = stringSampler(sampleLen)
    const batchSize = 128

    await net.train(dataset, batchSize, sampleLen)
}

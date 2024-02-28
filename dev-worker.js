import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    console.log(event.data)

    const net = new ODE({
        backend: 'cpu',
        layout: [128, 128],
        learningRate: 1e-2,
        decayRate: 0.9,
        momentum: 0,
        epsilon: 1e-8,
        predictLength: 100,
        inputLength: 300,
        embeddingDimensions: 256
    })
    await net.init()

    console.log(net.model.summary())

    const sampleLen = 64
    const dataset = stringSampler(sampleLen)
    const batchSize = 64

    await net.train(dataset, batchSize)
}

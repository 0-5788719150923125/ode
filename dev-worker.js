import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    console.log(event.data)

    const net = new ODE({
        backend: 'cpu',
        layout: [48, 48, 48],
        learningRate: 1e-2,
        predictLength: 100,
        inputLength: 300,
        embeddingDimensions: 256
    })
    await net.init()

    console.log(net.model.summary())

    const sampleLen = 256
    const dataset = stringSampler(sampleLen)
    const batchSize = 16

    await net.train(dataset, batchSize)
}

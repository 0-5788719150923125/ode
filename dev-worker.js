import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

onmessage = async function (event) {
    console.log(event.data)

    const net = new ODE({
        layout: [128, 128, 128],
        learningRate: 1e-3,
        predictLength: 100,
        inputLength: 300,
        embeddingDimensions: 256
    })

    console.log(net.model.summary())

    const sampleLen = 30
    const dataset = stringSampler(sampleLen)
    const batchSize = 2

    await net.train(dataset, batchSize)
}

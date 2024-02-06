import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

const net = new ODE({
    layout: [64, 64, 64],
    learningRate: 1e-3,
    predictLength: 100,
    inputLength: 300,
    embeddingDimensions: 128
})

console.log(net.model.summary())

const sampleLen = 600
const dataset = stringSampler(sampleLen)
const batchSize = 128
await net.train(dataset, batchSize)

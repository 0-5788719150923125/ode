import fs from 'fs'
import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

const net = new ODE({
    layout: [128, 128, 128],
    learningRate: 1e-3,
    predictLength: 100,
    inputLength: 180,
    embeddingDimensions: 128
})

console.log(net.model.summary())

const batchSize = 128
const sampleLen = 180
await net.train(stringSampler(sampleLen), batchSize)

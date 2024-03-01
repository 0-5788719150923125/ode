import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

export async function trainModel() {
    const net = new ODE({
        backend: 'tensorflow',
        layout: [48, 48, 48],
        learningRate: 1e-5,
        decay: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 32
    })
    await net.init()

    console.log(net.model.summary())

    const batchSize = 64
    const gradientAccumulationSteps = 2
    const sampleLen = 128
    const generateEvery = 32

    const dataset = stringSampler(sampleLen)

    await net.train(
        dataset,
        batchSize,
        gradientAccumulationSteps,
        sampleLen,
        generateEvery
    )
}

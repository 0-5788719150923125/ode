import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

export async function trainModel() {
    const net = new ODE({
        backend: 'tensorflow',
        layout: [128],
        learningRate: 1e-5,
        decayRate: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 64
    })
    await net.init()

    console.log(net.model.summary())

    const batchSize = 32
    const gradientAccumulationSteps = 4
    const sampleLen = 128
    const generateEvery = 128

    const dataset = stringSampler(sampleLen)

    await net.train(
        dataset,
        batchSize,
        gradientAccumulationSteps,
        sampleLen,
        generateEvery
    )
}

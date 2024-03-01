import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

export async function trainModel() {
    const net = new ODE({
        backend: 'tensorflow',
        layout: [128, 128],
        learningRate: 1e-2,
        decay: 0,
        momentum: 0,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 16
    })
    await net.init()

    console.log(net.model.summary())

    const batchSize = 64
    const gradientAccumulationSteps = 2
    const sampleLen = 60
    const generateEvery = 16

    const dataset = stringSampler(sampleLen)

    await net.train(
        dataset,
        batchSize,
        gradientAccumulationSteps,
        sampleLen,
        generateEvery
    )
}

import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

export async function trainModel() {
    const net = new ODE({
        backend: 'tensorflow',
        layout: [128, 128, 128],
        learningRate: 1e-3,
        decay: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 16
    })
    await net.init()

    const trainArgs = {
        batchSize: 64,
        gradientAccumulationSteps: 2,
        sampleLen: 128,
        generateEvery: 32
    }

    const dataset = stringSampler(trainArgs.sampleLen)

    await net.train(dataset, trainArgs)
}

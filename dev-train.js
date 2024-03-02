import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

export async function trainModel(backend = 'cpu') {
    const trainArgs = {
        batchSize: 64,
        gradientAccumulationSteps: 2,
        sampleLen: 64,
        generateEvery: 64
    }

    const net = new ODE({
        backend: backend,
        layout: [128, 128],
        learningRate: 1e-2,
        decay: 0.9,
        momentum: 0.01,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 16,
        maxSequenceLength: trainArgs.sampleLen
    })

    await net.init()

    const dataset = stringSampler(trainArgs.sampleLen)

    await net.train(dataset, trainArgs)
}

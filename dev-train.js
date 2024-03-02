import ODE, { stringSampler } from './src/index.js'

export async function trainModel(backend = 'cpu') {
    const trainArgs = {
        batchSize: 64,
        gradientAccumulationSteps: 2,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 50
    }

    const net = new ODE({
        backend: backend,
        layout: [96, 96, 96],
        learningRate: 1e-2,
        decay: 9e-1,
        momentum: 1e-2,
        epsilon: 1e-8,
        predictLength: 100,
        embeddingDimensions: 32,
        maxSequenceLength: trainArgs.sampleLen
    })

    await net.init()

    const dataset = stringSampler(trainArgs.sampleLen)

    await net.train(dataset, trainArgs)
}

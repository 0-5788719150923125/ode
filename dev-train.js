import ODE, { stringSampler } from './src/index.js'

export async function trainModel(args) {
    const trainArgs = {
        backend: 'cpu',
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 8,
        ...args
    }

    const net = new ODE({
        backend: trainArgs.backend,
        layout: [64, 64, 64],
        learningRate: 1e-3,
        decay: 9e-1,
        momentum: 1e-2,
        epsilon: 1e-8,
        embeddingDimensions: 32,
        contextLength: trainArgs.sampleLen,
        maxSequenceLength: trainArgs.sampleLen
    })

    await net.init()

    const dataset = stringSampler(trainArgs.sampleLen)

    await net.train(dataset, trainArgs)
}

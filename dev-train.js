import ODE, { stringSampler } from './src/index.js'

export async function trainModel(args) {
    const trainArgs = {
        backend: 'cpu',
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 50,
        ...args
    }

    const net = new ODE({
        learningRate: 0.01,
        decay: 0.9,
        momentum: 0.1,
        epsilon: 1e-8,
        backend: trainArgs.backend,
        contextLength: trainArgs.sampleLen,
        maxSequenceLength: trainArgs.sampleLen,
        ...trainArgs
    })

    await net.init()

    const dataset = stringSampler(trainArgs.sampleLen)

    await net.train(dataset, trainArgs)
}

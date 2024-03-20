import ODE from './src/index.js'

export async function trainModel(args) {
    const trainArgs = {
        backend: 'cpu',
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 32,
        ...args
    }

    const net = await ODE({
        // learningRate: 0.001,
        // decay: 0.9,
        // momentum: 0.01,
        // epsilon: 1e-8,
        version: 3,
        backend: trainArgs.backend,
        contextLength: trainArgs.sampleLen,
        clipValue: 1.0,
        ...trainArgs
    })

    await net.init()

    const dataset = net.sampler('string')(
        trainArgs.sampleLen * 5,
        trainArgs?.overfit
    )

    await net.train(dataset, trainArgs)
}

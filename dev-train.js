import ODE from './src/index.js'

export async function trainModel(args) {
    const trainArgs = {
        backend: 'cpu',
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLength: 64,
        generateEvery: 64,
        predictLength: 64,
        ...args
    }

    const net = await ODE({
        version: 4,
        // learningRate: 0.001,
        // decay: 0.9,
        // momentum: 0.01,
        // epsilon: 1e-8,
        contextLength: trainArgs.sampleLength,
        clipValue: 1.0,
        ...trainArgs
    })

    await net.init()
    // await net.load()
    await net.tokenizer.writeVocabularyToFile()

    const dataset = net.sampler('string')(
        trainArgs.sampleLength * 5,
        trainArgs?.overfit
    )

    await net.train(dataset, trainArgs)
}

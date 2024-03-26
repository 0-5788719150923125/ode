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
        contextLength: trainArgs.sampleLength,
        clipValue: 1.0,
        ...trainArgs
    })

    await net.init()
    // await net.load()
    await net.tokenizer.writeVocabularyToFile()

    // const dataset = net.ode.samplers.stringSampler(
    //     trainArgs.sampleLength * 5,
    //     trainArgs?.overfit
    // )
    const dataset = net.ode.samplers.directorySampler(
        trainArgs.sampleLength * 5,
        trainArgs?.overfit,
        '/home/crow/Repos/vtx/lab/phi/train',
        '\n\n\n'
    )

    await net.train(dataset, trainArgs)
}

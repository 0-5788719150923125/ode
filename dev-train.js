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

    const gun = net.ode.samplers.gunSampler()
    await gun.init()
    // await gun.subscribeChannel('trade')
    // await gun.putDataset('phi', null)
    await gun.uploadDirectory('phi', '/home/crow/Repos/vtx/lab/phi/train')
    const dataset = await net.ode.samplers.stringSampler(
        trainArgs.sampleLength * 5,
        trainArgs?.overfit,
        await gun.getDataset('phi') // remove this to just use the default Shakespeare dataset
    )

    await net.train(dataset, trainArgs)
}

import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'

export async function trainModel(dataGenerator) {
    // XXX: .
    const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
    const sampleStep = 3 // number of characters to jump between segments of input text
    const epochs = 150 // the total number of times to update the training weights
    const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data

    const seed = ''
    for (let i = 0; i < epochs; ++i) {
        const ds = tf.data.generator(
            createBatchGenerator(dataGenerator, this.characters)
        )
        await this.model.fitDataset(ds, {
            epochs: 1,
            batchSize,
            // validationSplit,
            callbacks: {
                onTrainBegin: () => console.log(`Epoch ${i + 1} of ${epochs}:`),
                onBatchEnd: async (batch, logs) => console.log(logs),
                onEpochEnd: async (epoch, logs) =>
                    console.log(await this.generate(seed, 0.7))
            }
        })
    }
}

function createBatchGenerator(dataGenerator, characters) {
    return function* () {
        yield* batchGenerator(dataGenerator, characters)
    }
}

function* batchGenerator(dataGenerator, characters) {
    console.log('trying to load batches')
    while (true) {
        const text = dataGenerator.next().value

        console.log(text)

        // Extract necessary parameters from text or context
        const textLength = text.length
        const sampleLen = 60 // Adjust as needed
        const sampleStep = 3 // Adjust as needed
        // const charSet = Array.from(new Set(Array.from(text)))
        // const charSetSize = charSet.length

        // Create tensors for the current batch
        const textIndices = new Uint16Array(
            Array.from(text).map((e) => characters.indexOf(e))
        )
        const trainingIndices = []

        for (let i = 0; i < textLength - sampleLen - 1; i += sampleStep) {
            trainingIndices.push(i)
        }

        tf.util.shuffle(trainingIndices)

        const xsBuffer = new tf.TensorBuffer([1, sampleLen, characters.length]) // One example per batch
        const ysBuffer = new tf.TensorBuffer([1, characters.length])

        const batchIndex = trainingIndices[0 % trainingIndices.length]
        for (let j = 0; j < sampleLen; ++j) {
            xsBuffer.set(1, 0, j, textIndices[batchIndex + j])
        }
        ysBuffer.set(1, 0, textIndices[batchIndex + sampleLen])

        yield { xs: xsBuffer.toTensor(), ys: ysBuffer.toTensor() }
    }
}

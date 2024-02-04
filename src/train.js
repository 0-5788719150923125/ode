import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'
import { randomBetween } from './common'

export async function trainModel(dataGenerator) {
    // XXX: .
    // const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
    // const sampleStep = 3 // number of characters to jump between segments of input text
    // const epochs = 150 // the total number of times to update the training weights
    // const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data

    const seed = ''
    // for (let i = 0; i < epochs; ++i) {
    const ds = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab)
    )
    await this.model.fitDataset(ds, {
        epochs: 1000,
        batchSize,
        // validationSplit,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => console.log(logs),
            onEpochEnd: async (epoch, logs) =>
                console.log(await this.generate(seed, 0.7))
        }
    })
    // }
}

function createBatchGenerator(dataGenerator, vocab) {
    return function* () {
        yield* batchGenerator(dataGenerator, vocab)
    }
}

function* batchGenerator(dataGenerator, vocab) {
    while (true) {
        const text = dataGenerator.next().value

        console.log(text)

        // Extract necessary parameters directly
        const textIndices = new Uint16Array(
            text.split('').map((e) => vocab.indexOf(e))
        )
        const sampleLen = 60 // Assuming a fixed sample length
        // const sampleLen = randomBetween(60, 300)

        // Create tensors directly for the single batch
        const xsBuffer = tf.buffer([1, sampleLen, vocab.length])
        const ysBuffer = tf.buffer([1, vocab.length])

        // Fill the tensors directly without intermediate arrays
        for (let i = 0; i < sampleLen; ++i) {
            xsBuffer.set(1, 0, i, textIndices[i])
        }
        ysBuffer.set(1, 0, textIndices[sampleLen])

        yield { xs: xsBuffer.toTensor(), ys: ysBuffer.toTensor() }
    }
}

import * as tf from '@tensorflow/tfjs-node-gpu'
// import '@tensorflow/tfjs-node'
// import * as tf from '@tensorflow/tfjs'
import { randomBetween } from './common'

export async function trainModel(dataGenerator) {
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data

    const seed = ''

    const ds = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab)
    )
    await this.model.fitDataset(ds, {
        epochs: 1000,
        batchSize,
        // validationSplit,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                if (batch % 128 === 0) {
                    console.log(logs)
                    console.log(await this.generate(seed, 0.7))
                }
            },
            onEpochEnd: async (epoch, logs) =>
                console.log(await this.generate(seed, 0.7))
        }
    })
}

function createBatchGenerator(dataGenerator, vocab) {
    return function* () {
        yield* batchGenerator(dataGenerator, vocab)
    }
}

function* batchGenerator(dataGenerator, vocab) {
    while (true) {
        const text = dataGenerator.next().value

        // console.log(text)

        // Extract necessary parameters directly
        const filteredText = text
            .split('')
            .filter((e) => vocab.indexOf(e) !== -1)
            .join('')
        const textIndices = new Uint16Array(
            filteredText.split('').map((e) => vocab.indexOf(e))
        )
        const sampleLength = textIndices.length - 1

        // Create tensors directly for the single batch
        const xsBuffer = tf.buffer([1, sampleLength, vocab.length])
        const ysBuffer = tf.buffer([1, vocab.length])

        // Fill the tensors directly without intermediate arrays
        for (let i = 0; i < sampleLength; ++i) {
            xsBuffer.set(1, 0, i, textIndices[i])
        }
        ysBuffer.set(1, 0, textIndices[sampleLength])

        yield { xs: xsBuffer.toTensor(), ys: ysBuffer.toTensor() }
    }
}

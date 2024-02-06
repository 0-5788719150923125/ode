import * as tf from '@tensorflow/tfjs-node-gpu'
// import '@tensorflow/tfjs-node'
// import * as tf from '@tensorflow/tfjs'

export async function trainModel(dataGenerator, batchSize = 256) {
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
                console.log(logs)
                for (let temp in [0, 0.3, 0.7, 0.9, 1.1]) {
                    const output = await this.generate('', temp)
                    console.log(output)
                }
            },
            onEpochEnd: async (epoch, logs) => console.log('epoch ended')
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
        const batch = []
        const batchSize = 64
        const maxSampleLength = 180
        for (let i = 0; i < batchSize; ++i) {
            const text = dataGenerator.next().value

            // Filter characters and convert to indices
            const filteredText = text
                .split('')
                .map((char) => vocab.indexOf(char))

            // Pad numeric indices to maximum length
            const paddedIndices = tf.pad(filteredText, [
                [0, maxSampleLength - filteredText.length]
            ])

            const xsBuffer = tf.buffer([maxSampleLength, vocab.length])
            const ysBuffer = tf.buffer([vocab.length])

            for (let j = 0; j < maxSampleLength; ++j) {
                xsBuffer.set(1, j, paddedIndices[j])
            }

            ysBuffer.set(1, paddedIndices[maxSampleLength])

            batch.push({
                xs: xsBuffer.toTensor(),
                ys: ysBuffer.toTensor()
            })
        }
        yield {
            xs: tf.stack(batch.map((sample) => sample.xs)),
            ys: tf.stack(batch.map((sample) => sample.ys))
        }
    }
}

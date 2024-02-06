import * as tf from '@tensorflow/tfjs-node'

export async function trainModel(dataGenerator, batchSize = 256) {
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data

    const seed = ''

    const ds = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab)
    )
    await this.model.fitDataset(ds, {
        epochs: 1,
        // batchSize,
        // validationSplit,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                console.log(logs)
                // does not work in Jest
                // if (batch === 3) {
                //     await this.saveModel()
                // }
                if (batch % 25 === 0) {
                    const output = await this.generate('', 0.7, 50)
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
    const batchSize = 64
    const sampleLen = 180 - 1 // Adjusted for input sequences
    const charSetSize = vocab.length

    while (true) {
        let xsArray = []
        let ysArray = []

        for (let i = 0; i < batchSize; ++i) {
            const text = dataGenerator.next().value

            // Convert characters to indices, filtering out characters not in vocab
            let textIndices = text
                .split('')
                .map((char) => vocab.indexOf(char))
                .filter((index) => index !== -1)

            // Ensure the sequence is not shorter than expected due to filtering
            if (textIndices.length < sampleLen) {
                // If the sequence is too short, pad it to the required length
                textIndices = [
                    ...textIndices,
                    ...Array(sampleLen - textIndices.length).fill(0)
                ] // Pad with zeros or another designated padding value
            }

            // Select a random start index for the sequence to add variability
            const startIdx = Math.floor(
                Math.random() * (textIndices.length - sampleLen)
            )

            // Create input sequence (xs) and target (ys)
            const xs = textIndices.slice(startIdx, startIdx + sampleLen)
            const ys = textIndices[startIdx + sampleLen] ?? 0 // Use nullish coalescing operator to handle undefined values

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(xsArray, [batchSize, sampleLen], 'int32')
        const ysTensor = tf.oneHot(tf.tensor1d(ysArray, 'int32'), charSetSize)

        yield { xs: xsTensor, ys: ysTensor }
    }
}

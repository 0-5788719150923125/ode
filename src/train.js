import * as tf from '@tensorflow/tfjs-node-gpu'

export async function trainModel(dataGenerator, batchSize = 256) {
    const alpha = 0.01 // Smoothing factor for EMA
    const emaCalc = emaGenerator(alpha)
    emaCalc.next() // Initialize the generator

    const ds = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab)
    )
    await this.model.fitDataset(ds, {
        epochs: 1,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                const updatedEma = emaCalc.next(logs.loss).value // Send new loss to generator and get updated EMA
                if (batch === 3) {
                    await this.saveModel()
                }
                if (batch % 5 === 0) {
                    console.log(`EMA=${updatedEma.toFixed(4)}`)
                }
                if (batch % 100 === 0) {
                    const output = await this.generate('', 0.7, 50)
                    console.log(logs)
                    console.log(output)
                }
            },
            onEpochEnd: async (epoch, logs) => console.log('epoch ended')
        }
    })
}

function* emaGenerator(alpha) {
    let ema = null // Initialize EMA to null
    while (true) {
        const newLoss = yield ema // Pause here and return 'ema'. When resumed, 'newLoss' gets the passed value.
        if (newLoss !== undefined) {
            // Check if 'newLoss' is provided
            ema = ema === null ? newLoss : alpha * newLoss + (1 - alpha) * ema // Update EMA with the new loss value
        }
    }
}

function createBatchGenerator(dataGenerator, vocab) {
    return function* () {
        yield* batchGenerator(dataGenerator, vocab)
    }
}

function* batchGenerator(dataGenerator, vocab) {
    const batchSize = 128
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
            // const startIdx = Math.floor(
            //     Math.random() * (textIndices.length - sampleLen)
            // )
            const startIdx = 0

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

import * as tf from '@tensorflow/tfjs'

export async function trainModel(
    dataGenerator,
    batchSize = 256,
    sampleLen = 256
) {
    const emaCalc = emaGenerator() // Initialize the generator
    emaCalc.next()

    const ds = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab, batchSize, sampleLen)
    )
    await this.model.fitDataset(ds, {
        epochs: 1,
        yieldEvery: 'auto',
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                const updatedEma = emaCalc.next(logs.loss).value // Send new loss to generator and get updated EMA
                // if (batch === 3) {
                //     await this.saveModel()
                // }
                console.log(`EMA=${updatedEma.toFixed(4)}`)
                if (batch % 25 === 0) {
                    const output = await this.generate('who', 0.23, 50)
                    console.log(logs)
                    console.log(output)
                }
            },
            onEpochEnd: async (epoch, logs) => console.log('epoch ended')
        }
    })
}

function* emaGenerator(alpha = 0.01) {
    let ema = null
    while (true) {
        const newLoss = yield ema // Pause here and return exponential moving average. When resumed, 'newLoss' gets the new value.
        if (newLoss !== undefined) {
            ema = ema === null ? newLoss : alpha * newLoss + (1 - alpha) * ema // Update EMA with the new loss value
        }
    }
}

function createBatchGenerator(dataGenerator, vocab, batchSize, inputLength) {
    return function* () {
        yield* batchGenerator(dataGenerator, vocab, batchSize, inputLength)
    }
}

function* batchGenerator(dataGenerator, vocab, batchSize, inputLength) {
    const sampleLength = inputLength // Adjusted for input sequences
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
            if (textIndices.length < sampleLength) {
                // If the sequence is too short, pad it to the required length
                textIndices = [
                    ...textIndices,
                    ...Array(sampleLength - textIndices.length).fill(0)
                ] // Pad with zeros or another designated padding value; 0 is bad here, since we've not implemented a pad token yet
            }

            // Create input sequence (xs) and target (ys)
            const xs = textIndices.slice(0, sampleLength)
            const ys = textIndices[sampleLength] ?? 0 // handle undefined values

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(
            xsArray,
            [batchSize, sampleLength],
            'int32'
        )
        const ysTensor = tf.oneHot(tf.tensor1d(ysArray, 'int32'), charSetSize)

        yield { xs: xsTensor, ys: ysTensor }
    }
}

import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-wasm'
// import '@tensorflow/tfjs-backend-webgpu'
// import '@tensorflow/tfjs-backend-webgl'

let currentBatch = null

export async function trainModel(
    dataGenerator,
    batchSize = 256,
    sampleLen = 256
) {
    // await tf.ready()
    // await tf.setBackend(this.config.backend || 'cpu')

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
                if (!currentBatch) return

                console.log(currentBatch)

                const updatedEma = emaCalc.next(logs.loss).value // Send new loss to generator and get updated EMA

                console.log(this.model.optimizer)
                // console.log(this.model.train.optimizer)
                // Gradient Clipping
                tf.tidy(() => {
                    const gradsAndVars = this.model.optimizer.computeGradients(
                        () => {
                            // Compute losses on the xs (input) data for this batch
                            console.log('doing predictions')
                            const predictions = this.model.predict(
                                currentBatch.xs
                            )
                            // .asType('float32')
                            console.log('getting loss')
                            const loss = this.model.compileArgs.loss(
                                predictions,
                                currentBatch.ys
                            )
                            return loss
                        }
                    )

                    // Apply gradient clipping (element-wise example)
                    console.log('clipping grads')
                    const clippedGradsAndVars = gradsAndVars.map(
                        ({ grad }) => ({
                            grad: tf.clipByValue(grad, -0.5, 0.5)
                        })
                    )

                    // Manually apply gradients (crucial step)
                    console.log('applying')
                    this.model.optimizer.applyGradients(clippedGradsAndVars)

                    // Release references to original gradients (optional, but good practice)
                    console.log('foreach')
                    gradsAndVars.forEach(({ grad }) => grad.dispose())
                })
                console.log(`EMA=${updatedEma.toFixed(4)}, LOSS=${logs.loss}`)
                if (batch % 100 === 0) {
                    console.log(logs)
                    for (const temp of [0.01, 0.1, 0.3, 0.7, 1.1]) {
                        const output = await this.generate('who', temp, 50)
                        console.log(`TEMPERATURE: ${temp}`)
                        console.log(output)
                    }
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
                ] // 0 is bad here, since we've not implemented a pad token yet
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

        currentBatch = { xs: xsTensor, ys: ysTensor }

        yield currentBatch
    }
}

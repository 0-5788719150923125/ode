import * as tf from '@tensorflow/tfjs'

let currentXs = null
let currentYs = null

export async function trainModel(
    dataGenerator,
    batchSize = 256,
    gradientAccumulationSteps = 1,
    sampleLen = 256,
    generateEvery = 32
) {
    let accumulatedGrads = {}
    let accumulationCounter = 0

    console.log(this.model.optimizer)

    const emaCalc = emaGenerator()
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
                const gradsAndVars = this.model.optimizer.computeGradients(
                    () => {
                        // Compute losses on the xs (input) data for this batch
                        const predictions = this.model
                            .predict(currentXs)
                            .asType('float32')

                        const loss = this.lossFunction(currentYs, predictions)

                        // // Reduce the loss tensor to a scalar if necessary
                        const lossScalar = loss.mean() // Use .mean(), .sum(), or another appropriate reduction

                        // // Extract the scalar loss value for logging
                        // const lossValue = lossScalar.dataSync()[0]
                        // console.log(lossValue)

                        return lossScalar
                    }
                )

                if (!gradsAndVars) return

                // currentXs.dispose()
                // currentYs.dispose()

                // Accumulate gradients
                Object.keys(gradsAndVars.grads).forEach((key) => {
                    if (!accumulatedGrads[key]) {
                        accumulatedGrads[key] = tf.keep(
                            tf.zerosLike(gradsAndVars.grads[key])
                        )
                    }
                    const tempGrad = tf.add(
                        accumulatedGrads[key],
                        gradsAndVars.grads[key]
                    )
                    accumulatedGrads[key].dispose()
                    accumulatedGrads[key] = tf.keep(tempGrad)
                })

                accumulationCounter++

                if (accumulationCounter === gradientAccumulationSteps) {
                    // Clip gradients to prevent explosion
                    const clippedGrads = {}
                    Object.keys(accumulatedGrads).forEach((key) => {
                        clippedGrads[key] = tf.keep(
                            tf.clipByValue(accumulatedGrads[key], -1.0, 1.0)
                        )
                        accumulatedGrads[key].dispose()
                    })

                    this.model.optimizer.applyGradients(clippedGrads)

                    // Reset for the next accumulation cycle
                    accumulatedGrads = {}
                    accumulationCounter = 0

                    // Dispose of the clipped gradients after application
                    Object.values(clippedGrads).forEach((tensor) =>
                        tensor.dispose()
                    )
                }

                // Ensure to dispose of grads after accumulation
                if (gradsAndVars && gradsAndVars.grads) {
                    Object.values(gradsAndVars.grads).forEach(
                        (grad) => grad && grad.dispose()
                    )
                }

                const updatedEma = emaCalc.next(logs.loss).value // Send new loss to generator and get updated EMA

                console.log(`EMA=${updatedEma.toFixed(4)}, LOSS=${logs.loss}`)
                if (batch % generateEvery === 0 && batch !== 0) {
                    console.log(logs)
                    for (const temp of [0.01, 0.1, 0.3, 0.7, 1.1]) {
                        const output = await this.generate('who', temp, 50)
                        console.log(`TEMPERATURE: ${temp}`)
                        console.log(output)
                    }
                }
            }
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

        currentXs = tf.clone(xsTensor)
        currentYs = tf.clone(ysTensor)

        yield { xs: xsTensor, ys: ysTensor }
    }
}

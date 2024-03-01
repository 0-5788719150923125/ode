import * as tfjs from '@tensorflow/tfjs'
import { randomBetween } from './utils.js'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

let currentXs = null
let currentYs = null

export async function trainModel(dataGenerator, args) {
    const { batchSize, gradientAccumulationSteps, sampleLen, generateEvery } =
        args

    let accumulatedGrads = {}
    let accumulationCounter = 0

    console.log(this.model.optimizer)

    const emaCalc = emaGenerator()
    emaCalc.next()

    const dataset = tf.data.generator(
        createBatchGenerator(dataGenerator, this.vocab, batchSize, sampleLen)
    )
    await this.model.fitDataset(dataset, {
        epochs: Number.MAX_SAFE_INTEGER,
        yieldEvery: 'auto',
        verbose: 0,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                const gradsAndVars = this.model.optimizer.computeGradients(
                    () => {
                        // Compute losses on the xs (input) data for this batch
                        const predictions = this.model
                            .predict(currentXs)
                            .asType('float32')

                        const loss = this.lossFunction(
                            currentYs,
                            predictions
                        ).mean()
                        return loss
                    }
                )

                if (!gradsAndVars) return

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

                    if (typeof window === 'undefined') {
                        await this.save()
                    }

                    for (const temp of [0, 0.1, 0.7]) {
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
        const newLoss = yield ema // Pause here and return exponential moving average
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
            // Ensure you get a length between 1 and sampleLength (inclusive)
            const randomLen = randomBetween(1, sampleLength)

            // Convert characters to indices, filtering out characters not in vocab
            let textIndices = text
                .split('')
                .map((char) => vocab.indexOf(char))
                .filter((index) => index !== -1)

            // If the sequence is too long, truncate it to randomLen
            if (textIndices.length > randomLen) {
                textIndices = textIndices.slice(0, randomLen)
            }

            // Pad sequences on the left if they are shorter than sampleLength
            if (textIndices.length < sampleLength) {
                textIndices = Array(sampleLength - textIndices.length)
                    .fill(vocab.indexOf('¶'))
                    .concat(textIndices)
            }

            // Create input sequence (xs)
            const xs = textIndices.slice(0, -1) // Exclude last character for input
            // Target (ys) is the last character of the sequence
            const ys = textIndices.slice(-1)[0] // Get last character index

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(
            xsArray,
            [batchSize, sampleLength - 1],
            'int32'
        )
        const ysTensor = tf.oneHot(tf.tensor1d(ysArray, 'int32'), charSetSize)

        currentXs = tf.clone(xsTensor)
        currentYs = tf.clone(ysTensor)

        yield { xs: xsTensor, ys: ysTensor }
    }
}

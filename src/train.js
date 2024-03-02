import * as tfjs from '@tensorflow/tfjs'
import {
    colors,
    elapsedTimeGenerator,
    findMatches,
    randomBetween
} from './utils.js'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

export async function startTraining(dataGenerator, args) {
    const trainArgs = {
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 50,
        ...args
    }

    let previousLoss = 0
    let accumulatedGrads = {}
    let accumulationCounter = 0
    let currentXs = null
    let currentYs = null
    const dataset = tf.data.generator(
        createBatchGenerator(
            dataGenerator,
            this.vocab,
            trainArgs.batchSize,
            trainArgs.sampleLen
        )
    )

    const timer = elapsedTimeGenerator()
    const emaCalc = emaGenerator()
    emaCalc.next()

    function createBatchGenerator(
        dataGenerator,
        vocab,
        batchSize,
        inputLength
    ) {
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
                        .fill(0)
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
            const ysTensor = tf.oneHot(
                tf.tensor1d(ysArray, 'int32'),
                charSetSize
            )

            currentXs = tf.clone(xsTensor)
            currentYs = tf.clone(ysTensor)

            yield { xs: xsTensor, ys: ysTensor }
        }
    }

    await this.model.fitDataset(dataset, {
        epochs: Number.MAX_SAFE_INTEGER,
        yieldEvery: 'auto',
        verbose: 0,
        callbacks: {
            onTrainBegin: () => {},
            onBatchEnd: async (batch, logs) => {
                // Compute losses on the xs (input) data for this batch
                const gradients = this.model.optimizer.computeGradients(() => {
                    const predictions = this.model.predict(currentXs)
                    const loss = this.model.loss[0](
                        currentYs,
                        predictions
                    ).mean()
                    return loss
                })

                // Accumulate gradients
                Object.keys(gradients.grads).forEach((key) => {
                    if (!accumulatedGrads[key]) {
                        accumulatedGrads[key] = tf.keep(
                            tf.zerosLike(gradients.grads[key])
                        )
                    }
                    const tempGrad = tf.add(
                        accumulatedGrads[key],
                        gradients.grads[key]
                    )
                    accumulatedGrads[key].dispose()
                    accumulatedGrads[key] = tf.keep(tempGrad)
                })

                accumulationCounter++

                if (
                    accumulationCounter === trainArgs.gradientAccumulationSteps
                ) {
                    // Clip gradients to prevent explosion
                    const clippedGrads = clipGradients(accumulatedGrads, 2.3)

                    // Reset for the next accumulation cycle
                    accumulatedGrads = {}
                    accumulationCounter = 0

                    this.model.optimizer.applyGradients(clippedGrads)

                    // Dispose of the clipped gradients after application
                    Object.values(clippedGrads).forEach((tensor) =>
                        tensor.dispose()
                    )
                }

                // Dispose of grads after accumulation
                Object.values(gradients.grads).forEach(
                    (grad) => grad && grad.dispose()
                )

                const updatedEma = emaCalc.next(logs.loss).value // Send new loss to generator and get updated EMA

                const comparedLoss = findMatches(
                    previousLoss.toFixed(14).toString(),
                    logs.loss.toFixed(14).toString()
                )
                previousLoss = logs.loss

                console.log(
                    `STEP=${batch}, EMA=${updatedEma.toFixed(4)}, LOSS=${comparedLoss.old}${colors.BLUE}${comparedLoss.new}${colors.WHITE}, ELAPSED=${timer.next().value / 1000}s`
                )
                if (
                    trainArgs.generateEvery > 0 &&
                    batch % trainArgs.generateEvery === 0 &&
                    batch !== 0
                ) {
                    console.log(logs)

                    if (typeof window === 'undefined') {
                        await this.save()
                    }

                    for (const temp of [0, 0.1, 0.7]) {
                        const prompt = dataGenerator
                            .next()
                            .value.slice(1, randomBetween(1, 16))
                        const output = await this.generate(
                            prompt,
                            temp,
                            trainArgs.predictLength
                        )
                        console.log(`TEMPERATURE: ${temp}`)
                        console.log(output)
                    }
                }
            }
        }
    })
}

function clipGradients(grads, value) {
    const clippedGrads = {}
    Object.keys(grads).forEach((key) => {
        clippedGrads[key] = tf.keep(tf.clipByValue(grads[key], -value, value))
        grads[key].dispose()
    })
    return clippedGrads
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

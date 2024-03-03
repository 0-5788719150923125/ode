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

    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        trainArgs.gradientAccumulationSteps
    )
    const timer = elapsedTimeGenerator()
    const emaCalc = emaGenerator()
    emaCalc.next()

    await this.model.fitDataset(dataset, {
        epochs: Number.MAX_SAFE_INTEGER,
        yieldEvery: 'auto',
        verbose: 0,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                // Compute losses and accumulate gradients
                gradientAccumulator.compute(currentXs, currentYs).step()

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
}

class GradientAccumulator {
    constructor(model, optimizer, accumulationSteps) {
        this.model = model
        this.optimizer = optimizer
        this.accumulationSteps = accumulationSteps
        this.accumulationCounter = 0
        this.accumulatedGrads = {}
    }

    compute(currentXs, currentYs) {
        this.gradients = computeGradients(
            this.model,
            this.optimizer,
            currentXs,
            currentYs
        )
        return this
    }

    async step() {
        Object.keys(this.gradients.grads).forEach((key) => {
            if (!this.accumulatedGrads[key]) {
                this.accumulatedGrads[key] = tf.keep(
                    tf.zerosLike(this.gradients.grads[key])
                )
            }
            const tempGrad = tf.add(
                this.accumulatedGrads[key],
                this.gradients.grads[key]
            )
            this.accumulatedGrads[key].dispose()
            this.accumulatedGrads[key] = tf.keep(tempGrad)
        })

        this.accumulationCounter++

        if (this.accumulationCounter === this.accumulationSteps) {
            // Clip gradients to prevent explosion
            const clippedGrads = clipGradients(this.accumulatedGrads, 2.3)

            // Reset for the next accumulation cycle
            this.accumulatedGrads = {}
            this.accumulationCounter = 0

            this.optimizer.applyGradients(clippedGrads)

            // Dispose of the clipped gradients after application
            Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
        }

        // Dispose of grads after accumulation
        Object.values(this.gradients.grads).forEach(
            (grad) => grad && grad.dispose()
        )
    }
}

function computeGradients(model, optimizer, currentXs, currentYs) {
    const gradients = optimizer.computeGradients(() => {
        const predictions = model.predict(currentXs)
        const loss = model.loss[0](currentYs, predictions).mean()
        return loss
    })
    return gradients
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

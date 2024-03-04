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

    let currentXs = null
    let currentYs = null

    const dataset = tf.data.generator(
        createBatchGenerator(
            dataGenerator,
            this.vocab,
            trainArgs.batchSize,
            trainArgs.sampleLen,
            trainArgs.predictLength
        )
    )

    const logger = new Logger()
    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        trainArgs.gradientAccumulationSteps
    )

    // The training loop
    await this.model.fitDataset(dataset, {
        epochs: Number.MAX_SAFE_INTEGER,
        yieldEvery: 'auto',
        verbose: 0,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                // Compute losses and accumulate gradients
                gradientAccumulator.compute(currentXs, currentYs).step()

                // Print logs
                logger.log(batch, logs.loss)

                // Print sample text
                await textSampler.call(
                    this,
                    batch,
                    dataGenerator,
                    trainArgs.generateEvery
                )
            }
        }
    })

    function createBatchGenerator(
        dataGenerator,
        vocab,
        batchSize,
        inputLength,
        predictLength
    ) {
        return function* () {
            yield* batchGenerator(
                dataGenerator,
                vocab,
                batchSize,
                inputLength,
                predictLength
            )
        }
    }

    function* batchGenerator(
        dataGenerator,
        vocab,
        batchSize,
        inputLength,
        predictLength
    ) {
        const sampleLength = inputLength - predictLength

        while (true) {
            let xsArray = []
            let ysArray = []

            for (let i = 0; i < batchSize; ++i) {
                const text = dataGenerator.next().value
                const sample = text.slice(
                    0,
                    randomBetween(1 + predictLength, inputLength)
                )

                // Convert characters to indices, filtering out characters not in vocab
                let textIndices = sample
                    .split('')
                    .map((char) => vocab.indexOf(char))
                    .filter((index) => index !== -1)

                // Ensure sequence is not longer than inputLength
                if (textIndices.length > inputLength) {
                    textIndices = textIndices.slice(0, inputLength)
                }

                // Pad sequences on the left if they are shorter than inputLength
                if (textIndices.length < inputLength) {
                    textIndices = Array(inputLength - textIndices.length)
                        .fill(0)
                        .concat(textIndices)
                }

                // Create input sequence (xs)
                let xs = textIndices.slice(0, sampleLength).map((char) => char)
                xs = Array(predictLength).fill(0).concat(xs)

                // Create target sequence (ys) and right-pad it to match inputLength
                let ys = textIndices.slice(sampleLength, inputLength)

                ys = ys.concat(Array(xs.length - predictLength).fill(0)) // Fill with padding value (0 or another designated value)

                xsArray.push(xs)
                ysArray.push(ys)
            }

            console.log(xsArray)
            console.log(ysArray)

            const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength])

            // Convert ysArray to a tensor, then one-hot encode
            const ysTensor = tf
                .oneHot(tf.tensor1d(ysArray.flat(), 'int32'), vocab.length)
                .reshape([batchSize, inputLength, vocab.length])

            currentXs = tf.clone(xsTensor)
            currentYs = tf.clone(ysTensor)

            yield { xs: xsTensor, ys: ysTensor }
        }
    }
}

class Logger {
    constructor() {
        this.timer = elapsedTimeGenerator()
        this.ema = emaGenerator()
        this.ema.next()
        this.previousLoss = 0
    }
    log(batch, currentLoss) {
        const updatedEma = this.ema.next(currentLoss).value // Send new loss to generator and get updated EMA

        const coloredLoss = findMatches(
            this.previousLoss.toFixed(14).toString(),
            currentLoss.toFixed(14).toString()
        )
        this.previousLoss = currentLoss

        console.log(
            `STEP=${batch}, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${colors.BLUE}${coloredLoss.new}${colors.WHITE}, ELAPSED=${this.timer.next().value / 1000}s`
        )
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

async function textSampler(batch, dataGenerator, generateEvery) {
    if (generateEvery > 0 && batch % generateEvery === 0 && batch !== 0) {
        if (typeof window === 'undefined') {
            await this.save()
        }

        for (const temp of [0, 0.1, 0.7]) {
            const prompt = dataGenerator
                .next()
                .value.slice(1, randomBetween(1, 16))
            const output = await this.generate(prompt, temp, 80)
            console.log(`TEMPERATURE: ${temp}`)
            console.log(output)
        }
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

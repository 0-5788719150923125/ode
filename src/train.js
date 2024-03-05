import * as tfjs from '@tensorflow/tfjs'
import {
    colors,
    elapsedTimeGenerator,
    findMatches,
    preprocessData,
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

    const logger = new Logger()
    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        trainArgs.gradientAccumulationSteps
    )

    let step = 0

    const dataset = batchGenerator(
        dataGenerator,
        this.vocab,
        trainArgs.batchSize,
        trainArgs.sampleLen,
        trainArgs.predictLength
    )

    // a custom train loop
    while (true) {
        step++

        const batch = dataset.next().value
        await gradientAccumulator.compute(batch.xs, batch.ys)
        await gradientAccumulator.step()

        // Print logs
        logger.log(step, gradientAccumulator.getLoss())

        // // Print sample text
        await textSampler.call(
            this,
            step,
            dataGenerator,
            trainArgs.generateEvery
        )
    }

    function* batchGenerator(
        dataGenerator,
        vocab,
        batchSize,
        inputLength,
        predictLength
    ) {
        while (true) {
            let xsArray = []
            let ysArray = []

            for (let i = 0; i < batchSize; ++i) {
                const text = dataGenerator.next().value
                const sample = text.slice(0, randomBetween(1, inputLength))

                const textIndices = preprocessData(
                    sample,
                    vocab,
                    inputLength + 1, // because we predict n + 1
                    'left'
                )

                // create input sequence
                const xs = textIndices.slice(0, -1)

                // predict the last character index
                const ys = textIndices.slice(-1)[0]

                xsArray.push(xs)
                ysArray.push(ys)
            }

            const xsTensor = tf.tensor2d(
                xsArray,
                [batchSize, inputLength],
                'int32'
            )
            const ysTensor = tf.oneHot(
                tf.tensor1d(ysArray, 'int32'),
                vocab.length
            )

            yield { xs: xsTensor, ys: ysTensor }
        }
    }
}

class Logger {
    constructor() {
        this.timer = elapsedTimeGenerator()
        this.ema = emaGenerator(0.1)
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

        const memory = tf.memory().numBytes / 1_000_000_000

        console.log(
            `STEP=${batch}, MEM=${memory.toFixed(4)}GB, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${colors.BLUE}${coloredLoss.new}${colors.WHITE}, ELAPSED=${this.timer.next().value / 1000}s`
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

    async compute(currentXs, currentYs) {
        const { value, grads, loss } = computeGradients(
            this.model,
            currentXs,
            currentYs
        )
        this.value = value
        this.gradients = grads
        this.loss = loss

        tf.dispose([currentXs, currentYs])
        return this
    }

    getLoss() {
        return this.loss
    }

    async step() {
        this.accumulationCounter++
        Object.keys(this.gradients).forEach((key) => {
            if (!this.accumulatedGrads[key]) {
                this.accumulatedGrads[key] = tf.keep(
                    tf.zerosLike(this.gradients[key])
                )
            }
            const tempGrad = tf.add(
                this.accumulatedGrads[key],
                this.gradients[key]
            )
            this.accumulatedGrads[key].dispose()
            this.accumulatedGrads[key] = tf.keep(tempGrad)
        })

        if (this.accumulationCounter === this.accumulationSteps) {
            // Average the gradients after accumulation
            Object.keys(this.accumulatedGrads).forEach((key) => {
                const avgGrad = this.accumulatedGrads[key].div(
                    tf.scalar(this.accumulationSteps)
                )
                this.accumulatedGrads[key].dispose()
                this.accumulatedGrads[key] = tf.keep(avgGrad)
            })

            // Clip gradients to prevent explosion
            const clippedGrads = clipGradients(this.accumulatedGrads, 2.3)

            // Reset for the next accumulation cycle
            this.accumulationCounter = 0
            Object.values(this.accumulatedGrads).forEach((tensor) =>
                tensor.dispose()
            )
            this.accumulatedGrads = {}

            this.optimizer.applyGradients(clippedGrads)

            // Dispose of the clipped gradients after application
            Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
        }

        // Dispose of grads after accumulation
        Object.values(this.gradients).forEach((grad) => grad && grad.dispose())
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
            const output = await this.generate(prompt, temp, 80, false)
            console.log(`TEMPERATURE: ${temp}`)
            console.log(output)
        }
    }
}

function computeGradients(model, currentXs, currentYs) {
    let loss
    const { value, grads } = tf.variableGrads(() => {
        const predictions = model.predict(currentXs)
        const lossValue = tf.losses.softmaxCrossEntropy(currentYs, predictions)
        loss = lossValue.dataSync()[0]
        return lossValue
    })
    return { value, grads, loss }
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

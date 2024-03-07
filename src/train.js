import * as tfjs from '@tensorflow/tfjs'
import {
    colors,
    elapsedTimeGenerator,
    emaGenerator,
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

    let step = 0
    const logger = new Logger()
    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        this.lossFunctions,
        trainArgs.gradientAccumulationSteps
    )

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
            trainArgs.generateEvery,
            trainArgs.predictLength
        )
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

        let memory = tf.memory()

        if (memory.numBytesInGPU) {
            memory = memory.numBytesInGPU / 1_000_000_000
        } else {
            memory = memory.numBytes / 1_000_000_000
        }

        console.log(
            `STEP=${batch}, MEM=${memory.toFixed(4)}GB, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${colors.BLUE}${coloredLoss.new}${colors.WHITE}, ELAPSED=${this.timer.next().value / 1000}s`
        )
    }
}

class GradientAccumulator {
    constructor(model, optimizer, lossFunctions, accumulationSteps) {
        this.model = model
        this.optimizer = optimizer
        this.lossFunctions = lossFunctions
        this.accumulationSteps = accumulationSteps
        this.accumulationCounter = 0
        this.accumulatedGrads = {}
    }

    async compute(currentXs, currentYs) {
        const { value, grads, loss } = computeGradients(
            this.model,
            this.lossFunctions[0],
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
        this.accumulatedGrads = accumulateGradients(
            this.gradients,
            this.accumulatedGrads
        )

        if (this.accumulationCounter === this.accumulationSteps) {
            // Average the gradients after accumulation
            this.accumulatedGrads = averageGradients(
                this.accumulatedGrads,
                this.accumulationSteps
            )

            // Clip gradients to prevent explosion
            const clippedGrads = clipGradients(this.accumulatedGrads, 2.3)

            // Reset for the next accumulation cycle
            this.accumulationCounter = 0
            Object.values(this.accumulatedGrads).forEach((tensor) =>
                tensor.dispose()
            )
            this.accumulatedGrads = {}

            // Update gradients, step the optimizer, changing weights
            this.optimizer.applyGradients(clippedGrads)

            // Dispose of the clipped gradients after application
            Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
        }

        // Dispose of grads after accumulation
        Object.values(this.gradients).forEach((grad) => grad && grad.dispose())
    }
}

function accumulateGradients(gradients, accumulatedGrads) {
    Object.keys(gradients).forEach((key) => {
        if (!accumulatedGrads[key]) {
            accumulatedGrads[key] = tf.keep(tf.zerosLike(gradients[key]))
        }
        const tempGrad = tf.add(accumulatedGrads[key], gradients[key])
        accumulatedGrads[key].dispose()
        accumulatedGrads[key] = tf.keep(tempGrad)
    })
    return accumulatedGrads
}

function averageGradients(grads, accumulationSteps) {
    const accumulatedGrads = grads
    Object.keys(accumulatedGrads).forEach((key) => {
        const avgGrad = accumulatedGrads[key].div(tf.scalar(accumulationSteps))
        accumulatedGrads[key].dispose()
        accumulatedGrads[key] = tf.keep(avgGrad)
    })
    return accumulatedGrads
}

function* batchGenerator(dataGenerator, vocab, batchSize, inputLength) {
    while (true) {
        let xsArray = []
        let ysArray = []

        for (let i = 0; i < batchSize; ++i) {
            const text = dataGenerator.next().value
            // const sample = text.slice(0, randomBetween(1, inputLength))
            const sample = text

            const textIndices = preprocessData(
                sample,
                vocab,
                inputLength, // because we predict n + 1
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
            [batchSize, inputLength - 1],
            'int32'
        )
        const ysTensor = tf.oneHot(tf.tensor1d(ysArray, 'int32'), vocab.length)

        yield { xs: xsTensor, ys: ysTensor }
    }
}

async function textSampler(batch, dataGenerator, generateEvery) {
    if (generateEvery > 0 && batch % generateEvery === 0 && batch !== 0) {
        if (typeof window === 'undefined') {
            await this.save()
        }

        for (const temp of [0, 0.3, 0.7]) {
            const prompt = dataGenerator
                .next()
                .value.slice(1, randomBetween(1, 16))
            const output = await this.generate(prompt, temp, 80, false)
            console.log(`TEMPERATURE: ${temp}`)
            console.log(output)
        }
    }
}

function computeGradients(model, lossFunction, currentXs, currentYs) {
    let loss
    const { value, grads } = tf.variableGrads(() => {
        const predictions = model.predict(currentXs)
        const lossValue = lossFunction(currentYs, predictions)
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

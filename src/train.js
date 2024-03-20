import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

let isBrowser = true
;(async function () {
    if (typeof window === 'undefined') {
        isBrowser = false
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-webgl'
import {
    colors,
    elapsedTimeGenerator,
    emaGenerator,
    findMatches,
    preprocessData,
    randomBetween
} from './utils.js'

export async function startTraining(dataGenerator, args) {
    const trainArgs = {
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLen: 64,
        generateEvery: 64,
        predictLength: 50,
        clipValue: 1.0,
        ...args
    }

    let step = 0
    const logger = new Logger()
    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        this.lossFunctions,
        trainArgs.gradientAccumulationSteps,
        trainArgs.clipValue
    )

    const dataset = batchGenerator(
        dataGenerator,
        this.tokenizer,
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

        // Print sample text
        await predictionSampler.call(
            this,
            step,
            dataGenerator,
            trainArgs.generateEvery,
            trainArgs.predictLength
        )
    }
}

class GradientAccumulator {
    constructor(model, optimizer, lossFunctions, accumulationSteps, clipValue) {
        this.model = model
        this.optimizer = optimizer
        this.lossFunctions = lossFunctions
        this.accumulationSteps = accumulationSteps
        this.clipValue = clipValue
        this.accumulationCounter = 0
        this.accumulatedGrads = {}
    }

    async compute(currentXs, currentYs) {
        const { grads, loss } = computeGradients(
            this.model,
            this.lossFunctions[0],
            currentXs,
            currentYs
        )

        this.gradients = grads
        this.loss = loss

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
            const clippedGrads = clipGradients(
                this.accumulatedGrads,
                this.clipValue
            )

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

function computeGradients(model, lossFunction, currentXs, currentYs) {
    let loss

    const { value, grads } = tf.tidy(() =>
        tf.variableGrads(() => {
            const predictions = model.predict(currentXs)
            const weights = null
            const smoothing = null
            const reduction = tf.Reduction.MEAN
            const lossValue = lossFunction(
                currentYs,
                predictions,
                weights,
                smoothing,
                reduction
            )
            loss = lossValue.dataSync()[0]
            return lossValue
        })
    )
    tf.dispose([currentXs, currentYs, value])
    return { grads, loss }
}

function clipGradients(grads, value) {
    const clippedGrads = {}
    Object.keys(grads).forEach((key) => {
        clippedGrads[key] = tf.keep(tf.clipByValue(grads[key], -value, value))
        grads[key].dispose()
    })
    return clippedGrads
}

function averageGradients(grads, accumulationSteps) {
    const divisor = tf.scalar(accumulationSteps) // Create the scalar outside the loop
    Object.keys(grads).forEach((key) => {
        const gradTensor = grads[key]
        const avgGrad = gradTensor.div(divisor)
        grads[key].dispose() // Dispose of the original gradient tensor
        grads[key] = avgGrad // Update with the averaged gradient
    })
    divisor.dispose()

    return grads
}

function accumulateGradients(gradients, accumulatedGrads) {
    Object.keys(gradients).forEach((key) => {
        if (!accumulatedGrads[key]) {
            accumulatedGrads[key] = tf.zerosLike(gradients[key])
        }
        const tempGrad = tf.add(accumulatedGrads[key], gradients[key])
        accumulatedGrads[key].dispose()
        accumulatedGrads[key] = tf.keep(tempGrad)
    })
    return accumulatedGrads
}

// function createDynamicLrScheduler(
//     warmupSteps,
//     initialLearningRate,
//     postWarmupLearningRate
// ) {
//     // Example usage:
//     const warmupSteps = 1000
//     const initialLearningRate = 0.0001
//     const postWarmupLearningRate = 0.001
//     let totalSteps = 0
//     const numBatchesPerEpoch = 64
//     return {
//         onBatchEnd: async (batch, logs) => {
//             totalSteps++
//             if (totalSteps < warmupSteps) {
//                 const newLr =
//                     initialLearningRate +
//                     ((postWarmupLearningRate - initialLearningRate) *
//                         totalSteps) /
//                         warmupSteps
//                 model.optimizer.setLearningRate(newLr)
//             }
//         },
//         onEpochEnd: async (epoch, logs) => {
//             if (totalSteps >= warmupSteps) {
//                 model.optimizer.setLearningRate(postWarmupLearningRate)
//             }
//         }
//     }
// }

// Assuming you have a model compiled with an optimizer
// const customLrScheduler = createDynamicLrScheduler(
//     warmupSteps,
//     initialLearningRate,
//     postWarmupLearningRate
// )

function* batchGenerator(dataGenerator, tokenizer, batchSize, inputLength) {
    while (true) {
        let xsArray = []
        let ysArray = []

        for (let i = 0; i < batchSize; ++i) {
            const sample = dataGenerator.next().value

            const textIndices = preprocessData(
                sample,
                tokenizer,
                inputLength + 1, // Including the next token to predict
                'left'
            )

            // Input sequence (excluding the last token for prediction)
            const xs = textIndices.slice(0, inputLength)

            // Output sequence (the entire sequence shifted by one position to the left)
            const ys = textIndices.slice(1)

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')

        // use this format with sparse loss functions
        // const ysTensor = tf.tensor2d(ysArray, [batchSize, inputLength], 'int32')

        const ysTensor = tf.tidy(() => {
            return tf
                .tensor2d(ysArray, [batchSize, inputLength], 'int32')
                .oneHot(tokenizer.getLength())
                .reshape([batchSize, inputLength, tokenizer.getLength()])
        })

        yield { xs: xsTensor, ys: ysTensor }
    }
}

async function predictionSampler(
    batch,
    dataGenerator,
    generateEvery,
    maxLength = 64
) {
    if (generateEvery > 0 && batch % generateEvery === 0 && batch !== 0) {
        let white = colors.WHITE
        let color = colors.BLUE

        if (isBrowser) {
            white = ''
            color = ''
        } else {
            await this.save()
        }

        const seedLength = randomBetween(16, maxLength - 16)
        const prompt = dataGenerator.next().value.slice(1, seedLength)

        for (const temp of [0, 0.3, 0.7]) {
            const startTime = performance.now()
            const output = await this.generate(prompt, temp, maxLength, false)
            const endTime = performance.now()
            console.log(
                `TEMPERATURE: ${temp}, RATE: ${(endTime - startTime) / (maxLength - seedLength)} ms/token`
            )
            console.log(
                color + prompt + white + output.slice(prompt.length, -1)
            )
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

        let white = colors.WHITE
        let color = colors.BLUE
        if (currentLoss > 20.0) color = colors.RED

        const coloredLoss = findMatches(
            this.previousLoss.toFixed(14).toString(),
            currentLoss.toFixed(14).toString()
        )
        this.previousLoss = currentLoss

        let memory = tf.memory()

        if (memory.numBytesInGPU) {
            memory = 'VRAM=' + (memory.numBytesInGPU / 1_000_000_000).toFixed(4)
        } else {
            memory = 'MEM=' + (memory.numBytes / 1_000_000_000).toFixed(4)
        }

        if (isBrowser) {
            white = ''
            color = ''
        }

        console.log(
            `STEP=${batch}, ${memory}GB, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${color}${coloredLoss.new}${white}, ELAPSED=${this.timer.next().value / 1000}s`
        )
    }
}

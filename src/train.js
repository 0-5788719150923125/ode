import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
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
        this.tokenizer,
        trainArgs.batchSize,
        trainArgs.sampleLen,
        trainArgs.predictLength
    )

    // function createBatchGenerator(
    //     dataGenerator,
    //     tokenizer,
    //     batchSize,
    //     sampleLen,
    //     predictLength
    // ) {
    //     return function* () {
    //         yield* batchGenerator(
    //             dataGenerator,
    //             tokenizer,
    //             batchSize,
    //             sampleLen,
    //             predictLength
    //         )
    //     }
    // }

    // const dataset = tf.data.generator(
    //     createBatchGenerator(
    //         dataGenerator,
    //         this.tokenizer,
    //         trainArgs.batchSize,
    //         trainArgs.sampleLen,
    //         trainArgs.predictLength
    //     )
    // )

    // await this.model.fitDataset(dataset, {
    //     epochs: 1000,
    //     verbose: 0,
    //     batchSize: 2,
    //     callbacks: {
    //         onBatchEnd: async (batch, logs) => {
    //             console.log(tf.memory())
    //             // tf.dispose([outside, outside.xs, outside.ys])
    //         }
    //     }
    // })

    // a custom train loop
    while (true) {
        step++

        const batch = dataset.next().value
        await gradientAccumulator.compute(batch.xs, batch.ys)
        await gradientAccumulator.step()
        tf.dispose([batch.xs, batch.ys])
        // console.log(tf.memory())

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

        let color = colors.BLUE
        if (currentLoss > 20.0) color = colors.RED

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
            `STEP=${batch}, MEM=${memory.toFixed(4)}GB, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${color}${coloredLoss.new}${colors.WHITE}, ELAPSED=${this.timer.next().value / 1000}s`
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
        // this.value = value
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
            // tf.dispose(this.accumulatedGrads)
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

// function averageGradients(grads, accumulationSteps) {
//     const accumulatedGrads = grads
//     Object.keys(accumulatedGrads).forEach((key) => {
//         // const avgGrad = accumulatedGrads[key].div(tf.scalar(accumulationSteps))
//         // accumulatedGrads[key].dispose()
//         // accumulatedGrads[key] = tf.keep(avgGrad)
//     })
//     // tf.dispose(grads)
//     return accumulatedGrads
// }
function averageGradients(grads, accumulationSteps) {
    const divisor = tf.scalar(accumulationSteps) // Create the scalar outside the loop to reuse
    Object.keys(grads).forEach((key) => {
        const gradTensor = grads[key]
        const avgGrad = gradTensor.div(divisor)
        grads[key].dispose() // Dispose of the original gradient tensor
        grads[key] = avgGrad // Update with the averaged gradient
    })
    divisor.dispose() // Dispose of the scalar tensor after the loop

    // No need for tf.keep() here as grads[key] = avgGrad assigns the reference to the grads object,
    // which should be kept outside of this function.
    return grads
}

function* batchGenerator(dataGenerator, tokenizer, batchSize, inputLength) {
    while (true) {
        let xsArray = []
        let ysArray = []

        for (let i = 0; i < batchSize; ++i) {
            const sample = dataGenerator.next().value

            // Assuming `preprocessData` prepares the data correctly
            // and returns a sequence of length `inputLength + 1`
            const textIndices = preprocessData(
                sample,
                tokenizer,
                inputLength + 1, // Including the next token to predict
                'left'
            )

            // Input sequence (excluding the last token for prediction)
            const xs = textIndices.slice(0, inputLength)

            // Output sequence (the entire sequence shifted by one position to the left)
            // This makes the model predict the next token at every position in the sequence
            const ys = textIndices.slice(1)

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')

        // Convert ysArray into a 2D tensor, then use tf.oneHot if needed.
        // Note that depending on your model output and loss function,
        // you might directly use integer labels in ysArray for efficiency.
        const ysOneHot = tf.tidy(() => {
            const ysTensor = tf.tensor2d(
                ysArray,
                [batchSize, inputLength],
                'int32'
            )
            return tf
                .oneHot(ysTensor.flatten(), tokenizer.getLength())
                .reshape([batchSize, inputLength, tokenizer.getLength()])
        })

        // If your model outputs logits and you're using a sparse categorical cross-entropy loss,
        // you can directly use ysTensor as is, without converting to one-hot encoding.
        // Otherwise, if your model ends with a softmax and you use categorical cross-entropy,
        // convert ysTensor to one-hot encoding:

        // tf.dispose([ysTensor])

        yield { xs: xsTensor, ys: ysOneHot } // Use `ys: ysOneHot` if one-hot encoding
    }
}

// function* batchGenerator(dataGenerator, tokenizer, batchSize, inputLength) {
//     while (true) {
//         let xsArray = []
//         let ysArray = []

//         for (let i = 0; i < batchSize; ++i) {
//             const sample = dataGenerator.next().value

//             const textIndices = preprocessData(
//                 sample,
//                 tokenizer,
//                 inputLength + 1, // because we predict n + 1
//                 'left'
//             )

//             // create input sequence
//             const xs = textIndices.slice(0, -1)

//             // predict the last character index
//             const ys = textIndices.slice(-1)[0]

//             xsArray.push(xs)
//             ysArray.push(ys)
//         }

//         const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')
//         const ysTensor = tf.oneHot(
//             tf.tensor1d(ysArray, 'int32'),
//             tokenizer.getLength()
//         )

//         yield { xs: xsTensor, ys: ysTensor }
//     }
// }

async function textSampler(batch, dataGenerator, generateEvery) {
    if (generateEvery > 0 && batch % generateEvery === 0 && batch !== 0) {
        if (typeof window === 'undefined') {
            await this.save()
        }

        const maxLength = 50
        const seedLength = randomBetween(3, 16)
        const prompt = dataGenerator.next().value.slice(1, seedLength)

        for (const temp of [0, 0.3, 0.7]) {
            const startTime = performance.now()
            const output = await this.generate(prompt, temp, maxLength, false)
            const endTime = performance.now()
            console.log(
                `TEMPERATURE: ${temp}, RATE: ${(endTime - startTime) / (maxLength - seedLength)} ms/token`
            )
            console.log(output)
        }
    }
}

function computeGradients(model, lossFunction, currentXs, currentYs) {
    let loss

    const { value, grads } = tf.tidy(() =>
        tf.variableGrads(() => {
            const predictions = model.predict(currentXs)
            const lossValue = lossFunction(currentYs, predictions)
            // tf.dispose([currentXs, currentYs])
            loss = lossValue.dataSync()[0]
            return lossValue
        })
    )
    tf.dispose([value])
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

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

    let currentXs = null
    let currentYs = null

    // const dataset = tf.data.generator(
    //     createBatchGenerator(
    //         dataGenerator,
    //         this.vocab,
    //         trainArgs.batchSize,
    //         trainArgs.sampleLen,
    //         trainArgs.predictLength
    //     )
    // )

    // const dataset = tf.data.generator(
    //     createBatchGenerator(
    //         dataGenerator,
    //         this.vocab,
    //         trainArgs.batchSize,
    //         trainArgs.sampleLen,
    //         trainArgs.predictLength
    //     )
    // )

    // The old training loop
    // await this.model.fitDataset(dataset, {
    //     epochs: Number.MAX_SAFE_INTEGER,
    //     yieldEvery: 'auto',
    //     verbose: 0,
    //     callbacks: {
    //         onBatchBegin: async (batch, logs) => {
    //             // Compute losses and accumulate gradients
    //             // gradientAccumulator.compute(currentXs, currentYs).step()
    //         },
    //         onBatchEnd: async (batch, logs) => {
    //             // Print logs
    //             logger.log(batch, logs.loss)

    //             // Print sample text
    //             await textSampler.call(
    //                 this,
    //                 batch,
    //                 dataGenerator,
    //                 trainArgs.generateEvery
    //             )
    //         }
    //     }
    // })

    // function createBatchGenerator(
    //     dataGenerator,
    //     vocab,
    //     batchSize,
    //     inputLength,
    //     predictLength
    // ) {
    //     return function* () {
    //         yield* batchGenerator(
    //             dataGenerator,
    //             vocab,
    //             batchSize,
    //             inputLength,
    //             predictLength
    //         )
    //     }
    // }

    const dataset = batchGenerator(
        dataGenerator,
        this.vocab,
        trainArgs.batchSize,
        trainArgs.sampleLen,
        trainArgs.predictLength
    )

    const logger = new Logger()
    const gradientAccumulator = new GradientAccumulator(
        this.model,
        this.model.optimizer,
        trainArgs.gradientAccumulationSteps
    )

    let step = 0

    // a custom train loop
    while (true) {
        dataset.next().value
        let loss
        // Gradient Calculation using tf.tidy for automatic memory cleanup
        tf.tidy(() => {
            // Compute gradients with respect to the model's variables
            const { value, grads } = tf.variableGrads(() => {
                const predictions = this.model.predict(currentXs)
                const lossValue = tf.losses.softmaxCrossEntropy(
                    currentYs,
                    predictions
                )
                loss = lossValue.dataSync()[0]
                return lossValue
            })

            // Apply gradients to update the model's weights
            this.model.optimizer.applyGradients(grads)
            // console.log(Math.random())
            // const lossFunc = () => {
            //     console.log('predicting')
            //     const predictions = this.model.predict(currentXs)
            //     const lossValue = tf.losses.softmaxCrossEntropy(
            //         currentYs,
            //         predictions
            //     )
            //     return lossValue
            // }

            // // Compute gradients
            // const grads = tf.grads(lossFunc)
            // // console.log(grads)

            // // Apply gradients to update the model's weights
            // this.model.optimizer.applyGradients(grads)
        })

        // await gradientAccumulator.compute(currentXs, currentYs)
        // await gradientAccumulator.step()
        // // Print logs
        step++
        // logger.log(step, gradientAccumulator.getLoss())
        logger.log(step, loss)
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

            currentXs = tf.clone(xsTensor)
            currentYs = tf.clone(ysTensor)

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
        const { grads, loss } = computeGradients(
            this.model,
            this.optimizer,
            currentXs,
            currentYs
        )
        this.gradients = grads
        this.loss = loss

        tf.dispose([currentXs, currentYs])
        return this
    }

    getLoss() {
        return this.loss
    }

    async step() {
        this.optimizer.applyGradients(this.gradients.grads)
        tf.dispose(this.gradients)
        // Object.keys(this.gradients.grads).forEach((key) => {
        //     if (!this.accumulatedGrads[key]) {
        //         this.accumulatedGrads[key] = tf.keep(
        //             tf.zerosLike(this.gradients.grads[key])
        //         )
        //     }
        //     const tempGrad = tf.add(
        //         this.accumulatedGrads[key],
        //         this.gradients.grads[key]
        //     )
        //     this.accumulatedGrads[key].dispose()
        //     this.accumulatedGrads[key] = tf.keep(tempGrad)
        // })

        // this.accumulationCounter++

        // if (this.accumulationCounter === this.accumulationSteps) {
        //     // Average the gradients after accumulation
        //     Object.keys(this.accumulatedGrads).forEach((key) => {
        //         const avgGrad = this.accumulatedGrads[key].div(
        //             tf.scalar(this.accumulationSteps)
        //         )
        //         this.accumulatedGrads[key].dispose()
        //         this.accumulatedGrads[key] = tf.keep(avgGrad)
        //     })

        //     // Clip gradients to prevent explosion
        //     const clippedGrads = clipGradients(this.accumulatedGrads, 2.3)

        //     // Reset for the next accumulation cycle
        //     this.accumulationCounter = 0
        //     Object.values(this.accumulatedGrads).forEach((tensor) =>
        //         tensor.dispose()
        //     )
        //     this.accumulatedGrads = {}

        //     this.optimizer.applyGradients(clippedGrads)

        //     // Dispose of the clipped gradients after application
        //     Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
        // }

        // // Dispose of grads after accumulation
        // Object.values(this.gradients.grads).forEach(
        //     (grad) => grad && grad.dispose()
        // )
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

// function computeGradients(model, optimizer, currentXs, currentYs) {
//     // let loss // Define loss outside of tf.tidy
//     const { gradients, loss } = tf.tidy(() => {
//         const lossFn = () => {
//             const predictions = model.predict(currentXs)
//             return tf.losses.softmaxCrossEntropy(currentYs, predictions).mean()
//         }

//         // Use tf.variableGrads to calculate the gradients
//         const grads = tf.variableGrads(lossFn)
//         // loss = grads.value
//         return { gradients: grads, loss: grads.value } // Only return the computed gradients
//     })

//     // The loss tensor is converted to a number before it is disposed of
//     const lossValue = loss.dataSync()[0]
//     loss.dispose() // Dispose of the loss tensor to free memory

//     return { grads: gradients, loss: lossValue }
// }

// function computeGradients(model, optimizer, currentXs, currentYs) {
//     let loss

//     // Forward pass
//     const preds = model.predict(currentXs)

//     // Compute loss
//     loss = tf.losses.softmaxCrossEntropy(currentYs, preds)

//     // Compute gradients
//     const gradients = tf.grad(tf.losses.softmaxCrossEntropy)(
//         currentXs,
//         currentYs
//     )

//     // Apply gradients
//     // optimizer.applyGradients(grads)
//     // const gradients = tf.tidy(() => {
//     //     return optimizer.computeGradients(() => {
//     //         const predictions = model.predict(currentXs)
//     //         loss = tf.losses.softmaxCrossEntropy(currentYs, predictions).mean()
//     //         return loss
//     //     })
//     // })
//     return { grads: gradients, loss: loss.dataSync()[0] }
// }

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

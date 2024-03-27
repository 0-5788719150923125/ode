import {
    colors,
    elapsedTimeGenerator,
    emaGenerator,
    findMatches,
    preprocessData,
    randomBetween
} from './utils.js'

let tf
let isBrowser = true

export async function startTraining(dataGenerator, args) {
    tf = this.tf
    isBrowser = this.isBrowser
    const trainArgs = {
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLength: 64,
        generateEvery: 64,
        predictLength: 50,
        clipValue: 1.0,
        mode: this.config.mode || 'timeDistributed',
        ...args
    }

    let batch = 0
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
        trainArgs.sampleLength,
        trainArgs.mode
    )

    // a custom train loop
    try {
        while (true) {
            batch++
            if (batch % trainArgs.gradientAccumulationSteps === 0) {
                step++
                // Set learning rate via schedule
                this.model.optimizer.learningRate =
                    this.schedulers[0].next().value
            }

            // Fetch data and compute gradients
            const tensors = dataset.next().value
            await gradientAccumulator.compute(tensors.xs, tensors.ys)
            await gradientAccumulator.step()

            // Print logs
            const loss = gradientAccumulator.getLoss()
            logger.log(batch, step, loss, this.model.optimizer.learningRate)

            // Print sample text
            await predictionSampler.call(
                this,
                batch,
                dataGenerator,
                trainArgs.generateEvery,
                trainArgs.predictLength
            )
        }
    } catch (err) {
        console.error(err, 'Oof!')
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
            const clippedGrads = tf.tidy(() => {
                return clipByGlobalNorm(this.accumulatedGrads, this.clipValue)
            })

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
            // const predictions = model.call(currentXs, { training: true })
            const predictions = model.call(currentXs)
            const weights = null
            const smoothing = null
            const reduction = tf.Reduction.MEAN
            const lossValue = lossFunction(
                currentYs,
                predictions[0],
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

function clipByValue(grads, value) {
    const clippedGrads = {}
    Object.keys(grads).forEach((key) => {
        clippedGrads[key] = tf.keep(tf.clipByValue(grads[key], -value, value))
        grads[key].dispose()
    })
    return clippedGrads
}

function l2Loss(tensor) {
    // https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    return tf.div(tf.sum(tf.square(tensor)), 2)
}

function globalNorm(tensors) {
    // https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/clip_ops.py#L242
    const halfSquaredNorms = []
    tensors.forEach((tensor, ti) => {
        halfSquaredNorms.push(l2Loss(tensor))
    })
    const halfSquaredNorm = tf.sum(tf.stack(halfSquaredNorms))
    const norm = tf.sqrt(
        tf.mul(halfSquaredNorm, tf.scalar(2.0, halfSquaredNorm.dtype))
    )
    return norm
}

function clipByGlobalNorm(tensors, clipNorm) {
    // https://github.com/kamalkraj/minGPT-TF/blob/master/mingpt/optimization.py
    // https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/clip_ops.py#L291-L382
    /*
    To perform the clipping, the values t_list[i] are set to:
        t_list[i] * clip_norm / max(global_norm, clip_norm)
    where:
        global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
    */
    const varNames = Object.keys(tensors)
    const tensorsArr = varNames.map((n) => tensors[n])
    const normalizedTensors = globalNorm(tensorsArr)
    const scale = tf.mul(
        clipNorm,
        tf.minimum(
            tf.div(tf.scalar(1.0), normalizedTensors),
            tf.div(tf.scalar(1.0, normalizedTensors.dtype), clipNorm)
        )
    )
    const tensorsClipped = []
    tensorsArr.forEach((tensor, ti) => {
        tensorsClipped.push(tf.clone(tf.mul(tensor, scale)))
    })
    const tensorsObjClipped = {}
    tensorsClipped.forEach((t, ti) => {
        tensorsObjClipped[varNames[ti]] = t
    })
    return tensorsObjClipped
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

function* batchGenerator(
    dataGenerator,
    tokenizer,
    batchSize,
    inputLength,
    mode = 'timeDistributed'
) {
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

            // Determine output sequence based on the mode
            let ys

            // Output sequence for singleLabel (just the next token)
            if (mode === 'oneLabel') {
                ys = [textIndices[inputLength]]
            }
            // Output sequence for timeDistributed (the entire sequence shifted by one position to the left)
            else {
                ys = textIndices.slice(1)
            }

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')

        const ysTensor = tf.tidy(() => {
            // Output labels should have a sequence length of just 1
            if (mode === 'oneLabel') {
                return tf.oneHot(
                    tf.tensor1d(ysArray.flat(), 'int32'),
                    tokenizer.getLength()
                )
            }
            // Output labels should match the length of xs sequences
            else {
                return tf
                    .tensor2d(ysArray, [batchSize, inputLength], 'int32')
                    .oneHot(tokenizer.getLength())
                    .reshape([batchSize, inputLength, tokenizer.getLength()])
            }
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

        for (const temp of [1, 0.7, 1.3]) {
            const startTime = performance.now()
            const output = await this.generate(prompt, temp, maxLength)
            const endTime = performance.now()
            console.log(
                `TEMPERATURE: ${temp}, RATE: ${((endTime - startTime) / (maxLength - seedLength)).toFixed(2)} ms/token`
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
        this.startTime = Date.now()
        this.totalElapsed = 0
        this.ema = emaGenerator()
        this.ema.next()
        this.previousLoss = 0
    }
    log(batch, step, currentLoss, learningRate) {
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

        const elapsed = this.timer.next().value
        this.totalElapsed += elapsed
        console.log(
            `STEP=${step}, BATCH=${batch}, ${memory}GB, EMA=${updatedEma.toFixed(4)}, LOSS=${coloredLoss.old}${color}${coloredLoss.new}${white}, LR=${learningRate.toFixed(5)}, ELAPSED=${(elapsed / 1000).toFixed(1)}s, TOTAL=${((Date.now() - this.startTime) / 1000 / 60 / 60).toFixed(3)}h`
        )
    }
}

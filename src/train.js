import {
    colors,
    elapsedTimeGenerator,
    emaGenerator,
    findMatches,
    getRandomBiasedNumber,
    preprocessData,
    randomBetween
} from './utils.js'

let tf

export async function trainModel(dataGenerator, args, extraCallbacks) {
    tf = this.tf
    const trainArgs = {
        batchSize: 32,
        gradientAccumulationSteps: 1,
        sampleLength: 64,
        generateEvery: 64,
        predictLength: 50,
        saveEvery: 0,
        clipValue: 1.0,
        labels: this.config.labels || 'timeDistributed',
        encoding: this.config.encoding || 'oneHot',
        sourceFormat: this.sourceFormat || 'text',
        imageSize: this.imageSize || 500,
        downsampling: this.downsampling?.rate || 1.0,
        ...args
    }

    this.batch = 0
    this.step = 0
    this.loss = 0

    const accumulator = new GradientAccumulator(
        this,
        trainArgs.gradientAccumulationSteps,
        trainArgs.clipValue
    )

    const callbacks = []
    for (const callback of extraCallbacks) {
        callbacks.push(new callback(this))
    }

    // a custom training loop
    while (true) {
        setLearningRate(
            this.batch,
            trainArgs.gradientAccumulationSteps,
            this.model,
            this.schedulers
        )

        this.batch++
        this.step = this.model.optimizer.iterations

        const data = await batchMaker(
            dataGenerator,
            this.tokenizer,
            trainArgs.batchSize,
            trainArgs.sampleLength,
            trainArgs.labels,
            trainArgs.encoding,
            trainArgs.sourceFormat,
            trainArgs.imageSize,
            trainArgs.downsampling
        )

        // if (trainArgs.downsampling !== 1.0) {
        //     const newTimeSteps = Math.floor(
        //         data.ys.shape[1] / trainArgs.downsampling
        //     )
        //     if (data.ys.shape[1] > newTimeSteps) {
        //         const newYs = tf.slice(
        //             data.ys,
        //             [0, data.ys.shape[1] - newTimeSteps, 0],
        //             [trainArgs.batchSize, newTimeSteps, data.ys.shape[2]]
        //         )
        //         data.ys.dispose()
        //         data.ys = newYs
        //     }
        // }

        // if (trainArgs.downsampling) {
        //     const newTimeSteps = Math.floor(data.ys.shape[1] / 2)
        //     if (data.ys.shape[1] > newTimeSteps) {
        //         const newYs = tf.slice(
        //             data.ys,
        //             [0, data.ys.shape[1] - newTimeSteps, 0],
        //             [trainArgs.batchSize, newTimeSteps, data.ys.shape[2]]
        //         )
        //         data.ys.dispose()
        //         data.ys = newYs
        //     }
        // }

        // Fetch data and compute gradients
        await accumulator.compute(data.xs, data.ys)
        await accumulator.step(this.step, this.batch)
        this.loss = accumulator.getLoss()

        for (const callback of callbacks) {
            await callback.step({
                batch: this.batch,
                step: this.step,
                loss: this.loss,
                dataGenerator,
                learningRate: this.model.optimizer.learningRate,
                ...trainArgs
            })
        }
    }
}

class GradientAccumulator {
    constructor(parent, accumulationSteps, clipValue) {
        this.parent = parent
        this.model = this.parent.model
        this.lossFunctions = this.parent.lossFunctions
        this.accumulationSteps = accumulationSteps
        this.clipValue = clipValue
        this.accumulationCounter = 0
        this.accumulatedGrads = {}
        this.currentStep = 0
        this.currentBatch = 0
    }

    async compute(currentXs, currentYs) {
        const { grads, loss } = computeGradients(
            this.model,
            this.lossFunctions,
            currentXs,
            currentYs,
            {
                training: true,
                step: this.currentStep,
                batch: this.currentBatch
            }
        )

        this.gradients = grads
        this.loss = loss
    }

    getLoss() {
        return this.loss
    }

    async step(step, batch) {
        this.currentStep = step
        this.currentBatch = batch
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

            // const filteredGrads = filterGradients.call(this, clippedGrads)

            // Update gradients, step the optimizer, changing weights
            // applyGradients(this.model, filteredGrads)
            // console.log(this.model.optimizer)
            this.model.optimizer.applyGradients(clippedGrads)

            Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
            // Object.values(filteredGrads).forEach((tensor) => tensor.dispose())
        }

        // Dispose of grads after accumulation
        Object.values(this.gradients).forEach((grad) => grad && grad.dispose())
    }
}

// Set learning rate via schedule
function setLearningRate(batch, gradientAccumulationSteps, model, schedulers) {
    if (batch % gradientAccumulationSteps === 0) {
        model.optimizer.learningRate = schedulers[0].next().value
    }
}

// function applyGradients(model, grads) {
//     const optimizer = model.optimizer
//     const trainableWeights = model.trainableWeights

//     const updatedGrads = {}
//     for (let i = 0; i < trainableWeights.length; i++) {
//         const weight = trainableWeights[i]
//         const grad = grads[weight.name]

//         if (grad != null && grad.shape.join(',') === weight.shape.join(',')) {
//             // console.log('grad ', grad.shape)
//             // console.log('weight ', weight.shape)
//             updatedGrads[weight.name] = grad
//         }
//     }

//     if (Object.keys(updatedGrads).length > 0) {
//         optimizer.applyGradients(updatedGrads)
//     }
// }

function computeGradients(
    model,
    lossFunctions,
    currentXs,
    currentYs,
    meta = { training: true }
) {
    let loss
    const lossFunction = lossFunctions[0].function
    const weights = lossFunctions[0].weights || null
    const smoothing = lossFunctions[0].smoothing || null
    const reduction = lossFunctions[0].reduction || tf.Reduction.MEAN
    const alpha = lossFunctions[0].alpha || null
    const gamma = lossFunctions[0].gamma || null
    const fromLogits = lossFunctions[0].fromLogits || true
    const { value, grads } = tf.tidy(() =>
        tf.variableGrads(() => {
            const predictions = model.call(currentXs, meta)

            let lossValue = lossFunction(
                currentYs,
                predictions[0],
                weights,
                smoothing,
                reduction,
                alpha,
                gamma,
                fromLogits
            )

            model.layers.forEach((layer) => {
                if (layer.hasOwnProperty('extraLoss')) {
                    lossValue = tf.add(lossValue, layer.extraLoss)
                }
                if (
                    layer.hasOwnProperty('latentMean') &&
                    layer.hasOwnProperty('latentLogVar')
                ) {
                    const klDivergence = tf.mul(
                        -0.5,
                        tf.mean(
                            layer.latentLogVar
                                .add(1)
                                .sub(layer.latentMean.square())
                                .sub(layer.latentLogVar.exp())
                        )
                    )
                    lossValue = tf.add(lossValue, klDivergence)
                }
            })

            loss = lossValue.dataSync()[0]
            return lossValue
        })
    )
    tf.dispose([currentXs, currentYs, value])
    return { grads, loss }
}

// function filterGradients(grads) {
//     const activeLayers = new Set()
//     const blockedLayers = new Set()

//     this.model.collectedTrainableWeights.forEach((variable) => {
//         if (variable.hasOwnProperty('trainable')) {
//             if (!variable.trainable) return blockedLayers.add(variable.name)
//         }
//         if (variable.hasOwnProperty('trainable_')) {
//             if (!variable.trainable_) return blockedLayers.add(variable.name)
//         }
//         activeLayers.add(variable.name)
//     })

//     const filteredGrads = {}
//     Object.keys(grads).forEach((varName) => {
//         const isActive = [...activeLayers].some((layerName) =>
//             varName.includes(layerName)
//         )

//         const isBlocked = [...blockedLayers].some((layerName) =>
//             varName.includes(layerName)
//         )

//         // If active, include this gradient in the filtered set
//         if (isActive && !isBlocked) {
//             filteredGrads[varName] = grads[varName]
//         }
//     })
//     // if (activeLayers.size > 0) console.log('active_layers', activeLayers)
//     // if (blockedLayers.size > 0) console.log('blocked_layers', blockedLayers)
//     return filteredGrads
// }

// function clipByValue(grads, value) {
//     const clippedGrads = {}
//     Object.keys(grads).forEach((key) => {
//         clippedGrads[key] = tf.keep(tf.clipByValue(grads[key], -value, value))
//         grads[key].dispose()
//     })
//     return clippedGrads
// }

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
    const divisor = tf.scalar(accumulationSteps)
    Object.keys(grads).forEach((key) => {
        const gradTensor = grads[key]
        const avgGrad = gradTensor.div(divisor)
        grads[key].dispose()
        grads[key] = avgGrad
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

async function batchMaker(
    dataGenerator,
    tokenizer,
    batchSize,
    inputLength,
    labels = 'timeDistributed',
    encoding = 'oneHot',
    sourceFormat = 'text',
    imageSize = 500,
    downsampling = 1.0
) {
    let xsArray = []
    let ysArray = []

    let sampleLength = inputLength

    for (let i = 0; i < batchSize; ++i) {
        // if (downsampling !== 1.0)
        //     sampleLength =
        //         inputLength - getRandomBiasedNumber(3, inputLength, 1.5)

        const sample = await dataGenerator.next().value.slice(0, sampleLength)

        const textIndices = preprocessData(
            sample,
            tokenizer,
            inputLength + 1, // Include the next token to predict
            'left'
        )

        // console.log(JSON.stringify(textIndices))

        // Input sequence (excluding the last token for prediction)
        let xs = textIndices.slice(0, inputLength)

        if (sourceFormat === 'image') {
            xs = tokenizer.getPixelData(tokenizer.decode(xs))
        }

        // Determine output sequence based on the mode
        let ys

        // Output sequence for singleLabel (just the next token)
        if (labels === 'oneLabel') {
            ys = [textIndices[inputLength]]
        }
        // Output sequence for timeDistributed (the entire sequence shifted by one position to the right)
        else {
            ys = textIndices.slice(1)
        }

        xsArray.push(xs)
        ysArray.push(ys)
    }

    let xsTensor
    if (sourceFormat === 'image') {
        xsTensor = tf.tensor4d(
            xsArray.flat(),
            [batchSize, imageSize, imageSize, 1],
            'float32'
        )
    } else {
        xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')
    }

    const ysTensor = tf.tidy(() => {
        if (encoding === 'integer') {
            // Output labels as integers
            if (mode === 'oneLabel') {
                return tf.tensor1d(ysArray.flat(), 'int32')
            } else {
                return tf.tensor2d(ysArray, [batchSize, inputLength], 'int32')
            }
        } else {
            // Output labels as one-hot encoded
            if (labels === 'oneLabel') {
                return tf.oneHot(
                    tf.tensor1d(ysArray.flat(), 'int32'),
                    tokenizer.getLength()
                )
            } else {
                return tf
                    .tensor2d(ysArray, [batchSize, inputLength], 'int32')
                    .oneHot(tokenizer.getLength())
                    .reshape([batchSize, inputLength, tokenizer.getLength()])
            }
        }
    })

    return { xs: xsTensor, ys: ysTensor }
}

export class ModelSaver {
    constructor(parent) {
        this.parent = parent
        this.savedAt = 0
    }

    async step(args) {
        if (
            args.saveEvery !== 0 &&
            args.step !== 0 &&
            args.step % args.saveEvery === 0 &&
            args.step !== this.savedAt
        ) {
            await this.parent.save()
            this.savedAt = args.step
        }
    }
}

export class PredictionSampler {
    constructor(parent) {
        this.parent = parent
    }

    async step(args) {
        if (
            args.generateEvery > 0 &&
            args.batch % args.generateEvery === 0 &&
            args.batch !== 0
        ) {
            const startTime = performance.now()
            const maxLength = args.predictLength

            const seedLength = randomBetween(16, maxLength - 16)
            const prompt = await args.dataGenerator
                .next()
                .value.slice(1, seedLength)

            const params = {
                doSample: true,
                temperature: 0.45,
                repetitionPenalty: 1.1,
                maxNewTokens: maxLength
            }
            const output = await this.parent.generate({
                prompt,
                ...params
            })
            const endTime = performance.now()
            console.log(
                `KWARGS: ${JSON.stringify(params)}, RATE: ${(
                    (endTime - startTime) /
                    (maxLength - seedLength)
                ).toFixed(2)} ms/token`
            )
            console.log(
                colors.BLUE +
                    prompt +
                    colors.WHITE +
                    output.slice(prompt.length, -1)
            )
        }
    }
}

export class ConsoleLogger {
    constructor(parent) {
        this.parent = parent
        this.timer = elapsedTimeGenerator()
        this.startTime = Date.now()
        this.totalElapsed = 0
        this.ema = emaGenerator()
        this.ema.next()
        this.previousLoss = 0
    }

    step(args) {
        const updatedEma = this.ema.next(args.loss).value // Send new loss to generator and get updated EMA

        let white = colors.WHITE
        let color = colors.BLUE
        if (args.loss > 20.0) color = colors.RED

        const coloredLoss = findMatches(
            this.previousLoss.toFixed(14).toString(),
            args.loss.toFixed(14).toString()
        )
        this.previousLoss = args.loss

        let memory = tf.memory()
        const numTensors = memory.numTensors
        memory = 'MEM=' + (memory.numBytes / 1_000_000_000).toFixed(4)

        const elapsed = this.timer.next().value
        this.totalElapsed += elapsed
        console.log(
            `STEP=${args.step}, BATCH=${args.batch}, EMA=${updatedEma.toFixed(
                4
            )}, LOSS=${coloredLoss.old}${color}${
                coloredLoss.new
            }${white}, LR=${args.learningRate.toFixed(
                5
            )}, ${memory}GB, TENSORS=${numTensors}, ELAPSED=${(
                elapsed / 1000
            ).toFixed(1)}s, TOTAL=${(
                (Date.now() - this.startTime) /
                1000 /
                60 /
                60
            ).toFixed(3)}h`
        )
    }
}

export class MetricsCollector {
    constructor(parent) {
        this.parent = parent
    }

    async step(stats) {
        await this.saveStatsToFile('./data/metrics.gg', {
            batch: stats.batch,
            step: stats.step,
            loss: stats.loss
        })
    }

    async saveStatsToFile(filename, stats) {
        if (!this.fs) {
            this.fs = await import('fs')
            this.fs.unlinkSync(filename)
        }
        const statsString = JSON.stringify(stats) + '\n'
        this.fs.appendFileSync(filename, statsString, 'utf8')
    }
}

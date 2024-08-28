import {
    colors,
    elapsedTimeGenerator,
    emaGenerator,
    findMatches,
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
        validateEvery: 0,
        predictLength: 50,
        saveEvery: 0,
        clipValue: 1.0,
        labels: this.labels || 'multiLabel',
        encoding: this.encoding || 'oneHot',
        sourceFormat: this.sourceFormat || 'text',
        imageSize: this.imageSize || 500,
        downsampling: this.downsampling?.rate || 1.0,
        ...args
    }

    this.batch = 0
    this.step = this.model.optimizer.step || 0
    this.loss = 0
    this.validationLoss = null

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
        await tf.nextFrame()
        setLearningRate(
            this.batch,
            trainArgs.gradientAccumulationSteps,
            this.model,
            this.schedulers
        )

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
            const r = await callback.step({
                batch: this.batch,
                step: this.step,
                loss: this.loss,
                valLoss: this.validationLoss,
                dataGenerator,
                tokenizer: this.tokenizer,
                learningRate: this.model.optimizer?.learningRate,
                lossFunctions: this.lossFunctions,
                ...trainArgs
            })
            if (r?.valLoss) {
                this.validationLoss = r.valLoss
            }
        }

        this.batch++
        if (this.batch % trainArgs.gradientAccumulationSteps === 0) {
            this.step++
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

            // Update gradients, step the optimizer, changing weights
            this.model.optimizer.applyGradients(clippedGrads)

            Object.values(clippedGrads).forEach((tensor) => tensor.dispose())
        }

        // Dispose of grads after accumulation
        Object.values(this.gradients).forEach((grad) => grad.dispose())
    }
}

// Set learning rate via schedule
function setLearningRate(batch, gradientAccumulationSteps, model, schedulers) {
    if (batch % gradientAccumulationSteps === 0) {
        model.optimizer.learningRate = schedulers[0].next().value
    }
}

function computeLoss(model, lossFunctions, labels, logits) {
    const lossFunction = lossFunctions[0].function
    const weights = lossFunctions[0].weights || null
    const smoothing = lossFunctions[0].smoothing || null
    const reduction = lossFunctions[0].reduction || tf.Reduction.MEAN
    const fromLogits = lossFunctions[0].fromLogits || true
    const alpha = lossFunctions[0].alpha || undefined
    const gamma = lossFunctions[0].gamma || undefined
    const sigma = lossFunctions[0].sigma || undefined
    const epsilon = lossFunctions[0].epsilon || undefined
    const q = lossFunctions[0].q || undefined

    let lossValue = lossFunction(
        labels,
        logits,
        weights,
        smoothing,
        reduction,
        fromLogits,
        alpha,
        gamma,
        sigma,
        epsilon,
        q
    )

    model.layers.forEach((layer) => {
        if (layer.hasOwnProperty('extraLoss')) {
            lossValue = tf.add(lossValue, layer.extraLoss)

            tf.dispose(layer.extraLoss)
            layer.extraLoss = null
        }
    })

    return lossValue
}

function computeGradients(
    model,
    lossFunctions,
    currentXs,
    currentYs,
    meta = { training: true }
) {
    let loss
    const { value, grads } = tf.tidy(() =>
        tf.variableGrads(() => {
            const predictions = model.call(currentXs, meta)

            let lossValue = computeLoss(
                model,
                lossFunctions,
                currentYs,
                predictions[0]
            )

            loss = lossValue.dataSync()[0]

            if (isNaN(loss)) {
                throw 'Loss values were NaN. Halting.'
            }

            return lossValue
        })
    )

    tf.dispose([currentXs, currentYs, value])

    return { grads, loss }
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
    labels = 'multiLabel',
    encoding = 'oneHot',
    sourceFormat = 'text',
    imageSize = 500,
    downsampling = 1.0,
    mode = 'train'
) {
    let xsArray = []
    let ysArray = []

    for (let i = 0; i < batchSize; ++i) {
        const sample = await dataGenerator.take({
            tokenizer,
            maxSeqLen: inputLength + 1,
            isValidating: mode === 'validation' ? true : false
        })

        const textIndices = preprocessData(
            sample,
            tokenizer,
            inputLength + 1, // Include the next token to predict
            'left'
        )

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
        this.lastStep = 0
    }

    async step(args) {
        if (
            args.generateEvery > 0 &&
            args.step % args.generateEvery === 0 &&
            this.lastStep !== args.step
        ) {
            const startTime = performance.now()
            const maxLength = args.predictLength

            const seedLength = randomBetween(16, maxLength - 16)
            const sample = await args.dataGenerator.take({
                tokenizer: args.tokenizer,
                maxSeqLen: seedLength
            })

            let prompt = sample
            if (Array.isArray(sample)) {
                prompt = args.tokenizer.decode(sample)
            }

            const params = {
                doSample: false,
                temperature: args.temperature,
                repetitionPenalty: args.repetitionPenalty,
                topK: args.topK,
                topP: args.topP,
                mirostat: args.mirostat,
                mirostatState: args.mirostatState,
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

export class ValidationHandler {
    constructor(parent) {
        this.parent = parent
        this.lastStep = 0
        this.loss = null
    }

    async step(args) {
        if (
            args.validateEvery > 0 &&
            args.step % args.validateEvery === 0 &&
            this.lastStep !== args.step
        ) {
            console.log('performing validation...')
            this.lastStep = args.step
            this.loss = 0

            let totalSteps = 0
            for (let i = 0; i <= args.validationSteps; i += args.batchSize) {
                const valData = await batchMaker(
                    args.dataGenerator,
                    args.tokenizer,
                    args.batchSize,
                    args.sampleLength,
                    args.labels,
                    args.encoding,
                    args.sourceFormat,
                    args.imageSize,
                    args.downsampling,
                    'validation'
                )

                tf.tidy(() => {
                    const predictions = this.parent.model.call(valData.xs, {
                        training: false
                    })

                    let lossValue = computeLoss(
                        this.parent.model,
                        args.lossFunctions,
                        valData.ys,
                        predictions[0]
                    )

                    this.loss += lossValue.dataSync()[0]
                })

                tf.dispose([valData.xs, valData.ys])
                totalSteps = totalSteps + args.batchSize
            }

            this.loss = this.loss / totalSteps
        }

        return { valLoss: this.loss }
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
        let valLoss = ''

        if (args.valLoss !== null) {
            valLoss = `VAL=${args.valLoss.toFixed(3)}, `
        }

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
            }${white}, ${valLoss}LR=${args.learningRate.toFixed(
                9
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
            try {
                this.fs.unlinkSync(filename)
            } catch (err) {
                // pass
            }
        }
        const statsString = JSON.stringify(stats) + '\n'
        this.fs.appendFileSync(filename, statsString, 'utf8')
    }
}

import losses from './losses.js'
import {
    LinearCongruentialGenerator,
    colors,
    elapsedTimeGenerator,
    emaGenerator,
    deterministicRandomString,
    findMatches,
    formatDate,
    preprocessData,
    randomBetween
} from './utils.js'

let tf

export async function trainModel(dataGenerator, args, extraCallbacks) {
    tf = this.tf
    const trainArgs = {
        batchSize: 32,
        gradientAccumulationSteps: 1,
        trainSteps: Infinity,
        sampleLength: 64,
        'oversample1.5x': 0.0,
        oversample2x: 0.0,
        oversample4x: 0.0,
        generateEvery: 64,
        validateEvery: 0,
        predictLength: 50,
        saveEvery: 0,
        clipValue: 1.0,
        ...args
    }

    this.batch = 0
    this.step = this.model.optimizer.step || 0
    this.loss = 0
    this.validationLoss = null
    this.validationPerplexity = null

    this.seed = trainArgs?.seed || null
    this.rng = {
        randomFloat: Math.random,
        randomBetween: randomBetween
    }
    if (this.seed !== null) {
        console.log(`trainer had a seed, using it: (${this.seed})`)
        const lcg = new LinearCongruentialGenerator(this.seed)
        this.rng = {
            randomFloat: (...args) => lcg.randomFloat(...args),
            randomBetween: (...args) => lcg.randomBetween(...args)
        }
    }

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

        let sampleLength = trainArgs.sampleLength
        let batchSize = trainArgs.batchSize
        if (this.rng.randomFloat() < trainArgs.oversample4x) {
            sampleLength = trainArgs.sampleLength * 4
            batchSize = Math.ceil(trainArgs.batchSize / 4)
        } else if (this.rng.randomFloat() < trainArgs.oversample2x) {
            sampleLength = trainArgs.sampleLength * 2
            batchSize = Math.ceil(trainArgs.batchSize / 2)
        } else if (this.rng.randomFloat() < trainArgs['oversample1.5x']) {
            sampleLength = Math.ceil(trainArgs.sampleLength * 1.5)
            batchSize = Math.ceil(trainArgs.batchSize / 2)
        }

        const data = await batchMaker(
            dataGenerator,
            this.tokenizer,
            batchSize,
            sampleLength,
            'train'
        )

        // Fetch data and compute gradients
        this.loss = await accumulator.step(this.step, this.batch, data)

        for (const callback of callbacks) {
            const r = await callback.step({
                batch: this.batch,
                step: this.step,
                loss: this.loss,
                valLoss: this.validationLoss,
                valPerplexity: this.validationPerplexity,
                dataGenerator,
                tokenizer: this.tokenizer,
                learningRate: this.model.optimizer?.learningRate,
                lossFunction: this.lossFunction,
                ...trainArgs
            })
            if (r?.valLoss) {
                this.validationLoss = r.valLoss
                this.validationPerplexity = r.valPerplexity
            }
        }

        this.batch++
        if (this.batch % trainArgs.gradientAccumulationSteps === 0) {
            this.step++
        }
        if (this.step >= trainArgs.trainSteps) {
            console.log(`Reached step ${this.step}. Halting.`)
            if (!isRunningInJest()) {
                this.save()
            }
            break
        }
    }
}

export const isRunningInJest = () => {
    return (
        process.env.JEST_WORKER_ID !== undefined ||
        process.env.NODE_ENV === 'test'
    )
}

class GradientAccumulator {
    constructor(parent, accumulationSteps, clipValue) {
        this.parent = parent
        this.model = this.parent.model
        this.lossFunction = this.parent.lossFunction
        this.accumulationSteps = accumulationSteps
        this.clipValue = clipValue
        this.accumulationCounter = 0
        this.accumulatedGrads = {}
        this.currentStep = 0
        this.currentBatch = 0
    }

    compute(currentXs, currentYs) {
        const { grads, loss } = computeGradients(
            this.parent,
            this.model,
            this.lossFunction,
            currentXs,
            currentYs,
            {
                training: true,
                step: this.currentStep,
                batch: this.currentBatch
            }
        )
        this.loss = loss
        return grads
    }

    async step(step, batch, data) {
        tf.tidy(() => {
            this.currentStep = step
            this.currentBatch = batch

            const newGrads = this.compute(data.xs, data.ys)

            if (this.accumulationCounter === 0) {
                Object.keys(newGrads).forEach((key) => {
                    const zeroTensor = tf.zerosLike(newGrads[key])
                    if (this.accumulatedGrads[key]) {
                        this.accumulatedGrads[key].assign(zeroTensor)
                    } else {
                        this.accumulatedGrads[key] = tf.variable(zeroTensor)
                    }
                })
            }

            this.accumulationCounter++

            accumulateGradients(this.accumulatedGrads, newGrads)

            if (this.accumulationCounter === this.accumulationSteps) {
                // Average the gradients after accumulation
                averageGradients(this.accumulatedGrads, this.accumulationSteps)

                // Clip gradients to prevent explosion
                const clippedGrads = clipByGlobalNorm(
                    this.accumulatedGrads,
                    this.clipValue
                )

                // Reset for the next accumulation cycle
                this.accumulationCounter = 0

                // Update gradients, step the optimizer, changing weights
                this.model.optimizer.applyGradients(clippedGrads)
            }
        })

        return this.loss
    }
}

// Set learning rate via schedule
function setLearningRate(batch, gradientAccumulationSteps, model, schedulers) {
    if (batch % gradientAccumulationSteps === 0) {
        model.optimizer.learningRate = schedulers[0].step()
    }
}

function computeLoss(model, lossFunctionArgs, labels, logits) {
    const lossFunction = losses[lossFunctionArgs.name]
    const weights = lossFunctionArgs.weights || null
    const smoothing = lossFunctionArgs.smoothing || null
    const reduction = lossFunctionArgs.reduction || tf.Reduction.MEAN
    const fromLogits = lossFunctionArgs.fromLogits || true
    const alpha = lossFunctionArgs.alpha || undefined
    const gamma = lossFunctionArgs.gamma || undefined
    const sigma = lossFunctionArgs.sigma || undefined
    const epsilon = lossFunctionArgs.epsilon || undefined
    const q = lossFunctionArgs.q || undefined

    const prediction = logits[0]

    // RNNs predict just a single label, so we slice the ys tensor here
    if (prediction.shape.length < 3) {
        labels = labels.squeeze().slice([labels.shape[1] - 1, 0], [1, -1])
    }

    let lossValue = lossFunction(
        labels,
        prediction,
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
    parent,
    model,
    lossFunction,
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
                lossFunction,
                currentYs,
                predictions
            )

            if (hasMethod(parent, 'postProcessing')) {
                const auxLoss = parent.postProcessing(predictions)
                lossValue = tf.add(lossValue, auxLoss)
            }

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

function hasMethod(instance, methodName) {
    return methodName in instance && typeof instance[methodName] === 'function'
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

function averageGradients(grads, divisor) {
    Object.keys(grads).forEach((key) => {
        grads[key].assign(grads[key].div(divisor))
    })
}

function accumulateGradients(accumulatedGrads, grads) {
    Object.keys(grads).forEach((key) => {
        const accGrad = tf.add(accumulatedGrads[key], grads[key])
        accumulatedGrads[key].assign(accGrad)
    })
}

async function batchMaker(
    dataGenerator,
    tokenizer,
    batchSize,
    inputLength,
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
        const xs = textIndices.slice(0, inputLength)

        // Shift right for labels, including last token
        const ys = textIndices.slice(1)

        xsArray.push(xs)
        ysArray.push(ys)
    }

    const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')

    const ysTensor = tf.tidy(() =>
        tf
            .tensor2d(ysArray, [batchSize, inputLength], 'int32')
            .oneHot(tokenizer.getLength())
            .reshape([batchSize, inputLength, tokenizer.getLength()])
    )

    return { xs: xsTensor, ys: ysTensor }
}

export class ModelSaver {
    constructor(parent) {
        this.parent = parent
        this.savedAt = 0
    }

    async step(args) {
        if (
            args.saveEvery === 0 ||
            args.step % args.saveEvery !== 0 ||
            this.savedAt === args.step
        ) {
            return
        }

        await this.parent.save()
        this.savedAt = args.step
    }
}

export class InferenceGenerator {
    constructor(parent) {
        this.parent = parent
        this.lastStep = 0
    }

    async step(args) {
        if (
            args.generateEvery === 0 ||
            args.step % args.generateEvery !== 0 ||
            this.lastStep === args.step
        ) {
            return
        }

        this.lastStep = args.step

        const seedLength = this.parent.rng.randomBetween(
            32,
            args.predictLength - 64
        )
        const sample = await args.dataGenerator.take({
            tokenizer: args.tokenizer,
            maxSeqLen: seedLength
        })

        let prompt = sample
        if (Array.isArray(sample)) {
            prompt = args.tokenizer.decode(sample)
        }

        const output = await this.parent.generate({
            ...args,
            prompt,
            maxNewTokens: args.predictLength
        })
        console.log(
            colors.BLUE +
                prompt +
                colors.WHITE +
                output.slice(prompt.length, -1)
        )
    }
}

export class ValidationHandler {
    constructor(parent) {
        this.parent = parent
        this.lastStep = 0
        this.clipValue = 25.0
    }

    async step(args) {
        if (
            args.validateEvery === 0 ||
            args.step % args.validateEvery !== 0 ||
            this.lastStep === args.step
        ) {
            return
        }

        console.log('performing validation...')
        args.dataGenerator.resetGenerator('validation')

        this.lastStep = args.step

        let totalLoss = 0
        let totalTokens = 0
        let totalBatches = 0

        for (let i = 0; i <= args.validationSteps; i++) {
            const valData = await batchMaker(
                args.dataGenerator,
                args.tokenizer,
                args.batchSize,
                args.sampleLength,
                'validation'
            )

            const [batchSize, seqLen, numFeatures] = valData.ys.shape

            tf.tidy(() => {
                const predictions = this.parent.model.call(valData.xs, {
                    training: false,
                    step: args.step,
                    batch: args.batch
                })

                const lossValue = computeLoss(
                    this.parent.model,
                    args.lossFunction,
                    valData.ys,
                    predictions
                )

                const numTokens = batchSize * seqLen
                const batchLoss = tf.clipByValue(lossValue, 0, this.clipValue)

                totalLoss += batchLoss.dataSync()[0] * numTokens
                totalTokens += numTokens
            })

            tf.dispose([valData.xs, valData.ys])

            totalBatches++
        }

        const valLoss = totalLoss / totalTokens
        const valPerplexity = Math.exp(valLoss)

        return { valLoss, valPerplexity }
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
        let valData = ''

        if (args.valLoss !== null) {
            valData = `VAL=${args.valLoss.toFixed(
                3
            )}, PPL=${args.valPerplexity.toFixed(3)}, `
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
            }${white}, ${valData}LR=${args.learningRate.toFixed(
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

function extractLayerInfo(model) {
    const layerInfoObject = {}

    model.layers.forEach((layer) => {
        const className = layer.constructor.name
        const layerName = layer.name

        // Split the layer name on the dash and get the prefix
        const [prefix] = layerName.split('-')

        // If this prefix hasn't been seen before, initialize it with a new Set
        if (!layerInfoObject[prefix]) {
            layerInfoObject[prefix] = new Set()
        }

        // Add the class name to the Set for this prefix
        layerInfoObject[prefix].add(className)
    })

    // Convert Sets to Arrays for easier handling if needed
    for (const prefix in layerInfoObject) {
        layerInfoObject[prefix] = Array.from(layerInfoObject[prefix])
    }

    return layerInfoObject
}

export class MetricsCollector {
    constructor(parent) {
        this.parent = parent
        this.state = {
            configuration: this.parent.config,
            tokenizer: this.parent.tokenizer.getConfig(),
            lossFunction: this.parent.lossFunction,
            optimizer: {
                name: this.parent.model.optimizer.constructor.name,
                ...this.parent.model.optimizer.getConfig()
            },
            scheduler: this.parent.schedulers[0].getConfig(),
            layers: extractLayerInfo(parent.model)
        }
        delete this.state.optimizer?.step
        this.runId = deterministicRandomString(JSON.stringify(this.state), 7)
        this.filename = './metrics.json'
        this.tempFilename = './metrics.tmp.json'
        this.metricsData = null
        this.buffer = []
        this.flushInterval = 5000 // 5 seconds
        this.maxBufferSize = 100
        this.historyLength = 250
        this.fs = null
        this.isFirstStep = true
    }

    appendToArray(array, item, maxItems) {
        const newArray = [...array]
        newArray.unshift(item)
        while (newArray.length > maxItems) {
            newArray.pop()
        }
        return newArray
    }

    async step(metrics) {
        if (this.fs === null) {
            const fsModule = await import('fs')
            this.fs = fsModule.promises
        }

        // Handle old metrics deletion or initialization only on the first step
        if (this.isFirstStep) {
            if (this.parent.wasResumed) {
                await this.loadExistingData()
            } else {
                await this.resetMetrics()
            }
            this.isFirstStep = false
        }

        const timestamp = Date.now()
        const logEntry = {
            runId: this.runId,
            timestamp,
            ...metrics
        }

        this.buffer.push(logEntry)

        await this.flush()
    }

    async flush() {
        if (this.buffer.length === 0) return
        if (this.buffer.length < this.maxBufferSize) return

        try {
            let data = this.metricsData || []

            for (const metrics of this.buffer) {
                const existingIndex = data.findIndex(
                    (entry) => entry.runId === this.runId
                )

                let existingEntry =
                    existingIndex !== -1 ? data[existingIndex] : {}

                const lastValidationLoss = existingEntry.validationLoss?.[0]
                const lastValidationPerplexity =
                    existingEntry.validationPerplexity?.[0]

                const filteredMetrics = {
                    runId: this.runId,
                    timestamp: metrics.timestamp,
                    date: formatDate(new Date(metrics.timestamp)),
                    batch: metrics.batch,
                    step: metrics.step,
                    version: metrics.version,
                    class: this.parent.constructor.name,
                    totalParams: this.parent.totalParams,
                    configuration: this.state.configuration,
                    tokenizer: this.state.tokenizer,
                    lossFunction: this.state.lossFunction,
                    optimizer: this.state.optimizer,
                    scheduler: this.state.scheduler,
                    layers: this.state.layers,
                    metricsInterval: this.maxBufferSize,
                    loss: metrics.loss,
                    validationLoss:
                        metrics.valLoss != null &&
                        metrics.valLoss !== lastValidationLoss
                            ? this.appendToArray(
                                  existingEntry.validationLoss || [],
                                  metrics.valLoss,
                                  this.historyLength
                              )
                            : existingEntry.validationLoss || [],
                    validationPerplexity:
                        metrics.valPerplexity != null &&
                        metrics.valPerplexity !== lastValidationPerplexity
                            ? this.appendToArray(
                                  existingEntry.validationPerplexity || [],
                                  metrics.valPerplexity,
                                  this.historyLength
                              )
                            : existingEntry.validationPerplexity || []
                }

                if (existingIndex !== -1) {
                    data[existingIndex] = filteredMetrics
                } else {
                    data.push(filteredMetrics)
                }
            }

            // Sort the data by timestamp, most recent first
            data.sort((a, b) => b.timestamp - a.timestamp)

            // Write to a temporary file first
            await this.fs.writeFile(
                this.tempFilename,
                JSON.stringify(data, null, 4),
                'utf8'
            )

            // Rename the temporary file to the actual filename
            await this.fs.rename(this.tempFilename, this.filename)

            this.metricsData = data

            // Clear the buffer after successful write
            this.buffer = []
        } catch (err) {
            // console.error(err)
        }
    }

    async loadExistingData() {
        try {
            const fileContent = await this.fs.readFile(this.filename, 'utf8')
            this.metricsData = JSON.parse(fileContent)
        } catch (error) {
            this.metricsData = []
        }
    }

    async resetMetrics() {
        try {
            const fileContent = await this.fs.readFile(this.filename, 'utf8')
            let data = JSON.parse(fileContent)

            // Filter out entries with the current runId
            data = data.filter((entry) => entry.runId !== this.runId)

            // Write the filtered data back to the file
            await this.fs.writeFile(
                this.filename,
                JSON.stringify(data, null, 4),
                'utf8'
            )

            console.log(`Reset metrics for this runId: ${this.runId}`)
            this.metricsData = data // Update the existingData
        } catch (error) {
            this.metricsData = [] // Initialize with an empty array if file doesn't exist
        }
    }

    async close() {
        await this.flush()
        try {
            // Attempt to remove the temporary file if it exists
            await this.fs.unlink(this.tempFilename)
        } catch (error) {
            // Ignore errors if the file doesn't exist
        }
    }
}

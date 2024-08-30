import losses from './losses.js'
import {
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
        sampleLength: 64,
        generateEvery: 64,
        validateEvery: 0,
        predictLength: 50,
        saveEvery: 0,
        clipValue: 1.0,
        labels: this.labels || 'multiLabel',
        encoding: this.encoding || 'oneHot',
        ...args
    }

    this.batch = 0
    this.step = this.model.optimizer.step || 0
    this.loss = 0
    this.validationLoss = null
    this.validationPerplexity = null

    const accumulator = new GradientAccumulator(
        this,
        trainArgs.gradientAccumulationSteps,
        trainArgs.clipValue
    )

    const callbacks = []
    for (const callback of extraCallbacks) {
        callbacks.push(new callback(this))
    }

    process.on('SIGINT', async () => {
        console.log('Received SIGINT. Gracefully stopping callbacks...')
        for (const callback of callbacks) {
            if (typeof callback.close !== 'undefined') {
                await callback.close()
            }
        }
        process.exit(0)
    })

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
            trainArgs.encoding
        )

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
    }
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

    async compute(currentXs, currentYs) {
        const { grads, loss } = computeGradients(
            this.model,
            this.lossFunction,
            currentXs,
            currentYs,
            this.parent.config.selfModel,
            this.parent.config.auxiliaryWeight,
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

function computeLoss(
    model,
    lossFunctionArgs,
    labels,
    logits,
    selfModel = false,
    auxiliaryWeight = 0.1
) {
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

    if (selfModel) {
        const hiddenStates = logits.slice(1)
        const selfModelingLoss = modelSelf(
            prediction,
            hiddenStates,
            auxiliaryWeight
        )
        lossValue = tf.add(lossValue, selfModelingLoss)
    }

    return lossValue
}

// https://arxiv.org/abs/2407.10188
function modelSelf(prediction, hiddenStates, auxiliaryWeight = 0.1) {
    let selfModelingLoss = 0

    hiddenStates.forEach((hiddenState) => {
        const loss = tf.tidy(() => {
            // Flatten the tensors
            const flat1 = prediction.reshape([-1])
            const flat2 = hiddenState.reshape([-1])

            // Determine the length to use (minimum of the two flattened tensors)
            const minLength = Math.min(flat1.shape[0], flat2.shape[0])

            // Slice the tensors to the common length
            const slice1 = flat1.slice([0], [minLength])
            const slice2 = flat2.slice([0], [minLength])

            return losses.meanSquaredError(slice1, slice2)
        })

        selfModelingLoss = tf.add(selfModelingLoss, loss)
    })

    return tf.mul(selfModelingLoss, auxiliaryWeight)
}

function computeGradients(
    model,
    lossFunction,
    currentXs,
    currentYs,
    selfModel = false,
    auxiliaryWeight = 0.1,
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
                predictions,
                selfModel,
                auxiliaryWeight
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

    const xsTensor = tf.tensor2d(xsArray, [batchSize, inputLength], 'int32')

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

export class ValidationHandler {
    constructor(parent) {
        this.parent = parent
        this.lastStep = 0
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

        this.lastStep = args.step

        let trueNLL = 0 // Negative Log Likelihood
        let totalNLL = 0
        let totalTokens = 0
        let totalSteps = 0

        let maxBatchSize = 0

        for (let i = 0; i <= args.validationSteps; i += args.batchSize) {
            const valData = await batchMaker(
                args.dataGenerator,
                args.tokenizer,
                args.batchSize,
                args.sampleLength,
                args.labels,
                args.encoding,
                'validation'
            )

            const [batchSize, seqLen, numFeatures] = valData.ys.shape

            maxBatchSize = Math.max(maxBatchSize, batchSize)

            if (batchSize < maxBatchSize) {
                console.log('batch size was wrong, returning')
                break
            }

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
                    predictions,
                    this.parent.config.selfModel,
                    this.parent.config.auxiliaryWeight
                )

                const numTokens = batchSize * seqLen
                const batchNLL = lossValue.dataSync()[0]

                trueNLL += batchNLL
                totalNLL += batchNLL * numTokens
                totalTokens += numTokens
            })

            tf.dispose([valData.xs, valData.ys])

            totalSteps += batchSize
        }

        const valLoss = trueNLL / totalSteps
        const averageNLL = totalNLL / totalTokens
        const valPerplexity = Math.exp(averageNLL)

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

export class MetricsCollector {
    constructor(parent) {
        this.parent = parent
        this.filename = './metrics.json'
        this.tempFilename = './metrics.tmp.json'
        this.runId = deterministicRandomString(
            JSON.stringify(this.parent.config),
            7
        )
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
                    configuration: this.parent.config,
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
                            : existingEntry.validationPerplexity || [],
                    metricsInterval: this.maxBufferSize,
                    lossFunction: metrics.lossFunction,
                    tokenizer: metrics.tokenizer?.model,
                    optimizer: {
                        name: this.parent.model.optimizer.constructor.name,
                        learningRate: metrics.learningRate,
                        weightDecay: this.parent.model.optimizer.weightDecay
                    }
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

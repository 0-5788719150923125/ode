import * as tfjs from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-webgpu'
import customLayers from './layers.js'
import customLosses from './losses.js'
import customOptimizers from './optimizers.js'
import customTokenizers from './tokenizers.js'
import customSchedulers from './schedulers.js'
import customSamplers from './samplers.js'
import Expert from './experts.js'
import './activations.js'
import './ops.js'
import { trainModel } from './train.js'
import { preprocessData } from './utils.js'

let tf = tfjs

/**
 * The base model class, which defines a standard API
 * that will remain compatible across all model versions.
 * @constructor
 * @param {Object} config - The configuration settings for the model.
 */
export default class ModelBase {
    constructor(config) {
        this.ode = {
            layers: customLayers,
            losses: customLosses,
            optimizers: customOptimizers,
            tokenizers: customTokenizers,
            schedulers: customSchedulers,
            samplers: customSamplers,
            expert: Expert
        }
        this.config = config
        this.model
        this.tokenizer
        this.contextLength = config.contextLength
    }

    async preInit() {
        this.tf = tfjs
        if (this.config.backend === 'tensorflow') {
            let backend = '@tensorflow/tfjs-node-gpu'
            tf = await import(backend)
            this.tf = tf
        }
        await this.tf.ready()
        await this.tf.setBackend(this.config.backend || 'cpu')
        this.defineTokenizer()
        if (typeof this.tokenizer.init === 'function') {
            await this.tokenizer.init()
        }
        this.defineLossFunctions()
    }

    async init() {
        await this.preInit()
        this.defineOptimizers()
        this.defineSchedulers()
        this.defineBuild()
        this.compile()
        this.postInit()
    }

    async load(type = 'file', path = `data/models/ode`) {
        await this.preInit()
        let modelPath = `${type}://${path}`
        if (type === 'file') modelPath += '/model.json'
        this.model = await this.tf.loadLayersModel(modelPath, {
            strict: true,
            streamWeights: true
        })

        // if a layer has experts
        for (const layer of this.model.layers) {
            if (layer.experts) {
                for (let i = 0; i < layer.numExperts; i++) {
                    const idx = i + 1
                    let expertPath = `${type}://${path}/experts/model${idx}`
                    if (type === 'file') expertPath += '/model.json'
                    layer.experts[i] = await this.tf.loadLayersModel(
                        expertPath,
                        {
                            strict: true,
                            streamWeights: true
                        }
                    )
                }
            }
        }
        console.log('successfully loaded model from disk')
        this.defineSchedulers()
        this.postInit()
    }

    async save(type = 'file', path = `data/models/ode`) {
        if (type === 'file') {
            const fs = await import('fs')
            fs.mkdirSync(path, { recursive: true })
        }
        await this.model.save(`${type}://${path}`, { includeOptimizer: true })
        // if a layer has experts
        for (const layer of this.model.layers) {
            if (layer.experts) {
                for (const expert of layer.experts) {
                    if (type === 'file') {
                        const fs = await import('fs')
                        fs.mkdirSync(`${path}/experts/${expert.name}`, {
                            recursive: true
                        })
                    }
                    await expert.save(
                        `${type}://${path}/experts/${expert.name}`,
                        { includeOptimizer: true }
                    )
                }
            }
        }
    }

    getStats() {
        const memory = this.tf.memory()
        const tensors = memory.numTensors

        let allocated = memory.numBytes / 1_000_000_000
        if (memory.numBytesInGPUAllocated) {
            allocated = memory.numBytesInGPUAllocated / 1_000_000_000
        }

        return {
            backend: this.tf.backend(),
            allocated: allocated,
            tensors,
            step: this.step,
            batch: this.batch,
            loss: this.loss,
            optimizer: this.model.optimizer.constructor.name
        }
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CharacterTokenizer()
    }

    defineLossFunctions() {
        this.lossFunctions = [
            {
                function: this.ode.losses.softmaxCrossEntropy,
                weights: null,
                smoothing: null,
                reduction: this.tf.Reduction.MEAN
            }
        ]
    }

    defineBuild() {
        throw 'Your model is missing a defineBuild() method. Did you forget to define one?'
    }

    defineOptimizers() {
        this.learningRate = 1e-3
        this.optimizers = [
            this.ode.optimizers.AdamW({
                learningRate: this.learningRate,
                weightDecay: 1e-2
            })
        ]
    }

    defineSchedulers() {
        this.learningRate = 1e-3
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    compile() {
        // this.model = enableGradientCheckpointing(this.model)
        this.model.compile({
            optimizer: this.optimizers[0],
            loss: this.lossFunctions[0].function
        })
    }

    postInit() {
        console.log(this.model.optimizer)
        console.log(this.model.summary())
        this.countParams()
        console.log(this.config)
        console.log(
            `\nTokenizer contains ${this.tokenizer.getLength()} tokens.`
        )
        console.log(`Loaded model: v${this.config.version}\n`)
    }

    countParams() {
        let layerParams = 0
        let expertParams = 0
        for (const layer of this.model.layers) {
            layerParams += layer.countParams()
            if (layer.experts) {
                for (const expert of layer.experts) {
                    expertParams += expert.countParams()
                }
            }
        }
        const oneMillion = 1_000_000
        console.log(
            (layerParams / oneMillion).toFixed(2) + 'M',
            'layer params,',
            (expertParams / oneMillion).toFixed(2) + 'M',
            'expert params'
        )
    }

    getWeightsByLayerPrefix(prefix = 'emb') {
        const layers = this.model.layers.filter((layer) =>
            layer.name.startsWith(prefix)
        )

        if (layers.length === 0) {
            console.warn('No embedding layers found.')
            return null
        }

        const weights = layers.map((layer) => {
            const layerWeights = layer.getWeights()
            return {
                layerName: layer.name,
                weights: layerWeights
            }
        })

        return weights
    }

    async generate({
        prompt = '',
        doSample = true,
        temperature = 0.7,
        topK = 0,
        topP = 1,
        repetitionPenalty = 1,
        maxNewTokens = 50,
        stopToken = null
    } = {}) {
        if (this.stateful) this.model.resetStates()
        const output = await generateText.call(this, {
            prompt,
            doSample,
            temperature,
            topK,
            topP,
            repetitionPenalty,
            maxNewTokens,
            stopToken
        })
        if (this.stateful) this.model.resetStates()
        return output
    }

    async train(dataGenerator, args, callbacks) {
        return await trainModel.call(this, dataGenerator, args, callbacks)
    }
}

async function generateText({
    prompt,
    doSample,
    temperature,
    topK,
    topP,
    repetitionPenalty,
    maxNewTokens,
    stopToken
} = {}) {
    const fixedLength = this.contextLength
    const isSingleLabel = this.model.outputs[0].shape.length === 2

    return this.tf.tidy(() => {
        if (!this.autoregressive) {
            return predictMany.call(this, { prompt })
        }

        let inputs
        if (isSingleLabel) {
            const tokenIndices = preprocessData(
                prompt,
                this.tokenizer,
                fixedLength,
                'left'
            )
            inputs = this.tf.tensor2d(tokenIndices, [1, fixedLength], 'int32')
        } else {
            inputs = prepareInputs.call(this, this.tokenizer.encode(prompt))
        }

        let tokenIndices = this.tokenizer.encode(prompt)

        let decodedText = prompt
        for (let step = 0; step < maxNewTokens; step++) {
            let indices = inputs
            if (this.sourceFormat === 'image') {
                const imageSize = this.imageSize
                indices = tf.tensor4d(
                    this.tokenizer.getPixelData(decodedText),
                    [1, imageSize, imageSize, 1],
                    'float32'
                )
            }

            const idxNext = predictOnce.call(
                this,
                indices,
                doSample,
                temperature,
                topK,
                topP,
                repetitionPenalty,
                isSingleLabel
            )
            // Append the predicted token index to the list of token indices
            tokenIndices.push(idxNext.dataSync()[0])

            // Decode the entire sequence of token indices to update generatedText
            decodedText = this.tokenizer.decode(tokenIndices)

            // Early stopping
            if (reachedStopToken(decodedText, stopToken)) {
                decodedText = decodedText.slice(0, -1)
                return decodedText
            }

            if (isSingleLabel) {
                inputs = inputs.slice([0, 1], [1, fixedLength - 1])
                const idxNextExpanded = idxNext.reshape([1, 1])
                inputs = this.tf.concat([inputs, idxNextExpanded], 1)
            } else {
                const idxNextExpanded = idxNext.expandDims(1)
                const idxNew = this.tf.concat([inputs, idxNextExpanded], 1)
                this.tf.dispose([inputs, idxNext, idxNextExpanded])
                inputs = idxNew
            }
        }
        this.tf.dispose([inputs])
        return decodedText
    })
}

function reachedStopToken(string, token) {
    return string.endsWith(token)
}

function predictOnce(
    idx,
    doSample,
    temperature,
    topK,
    topP,
    repetitionPenalty,
    isSingleLabel
) {
    return tf.tidy(() => {
        let logits
        if (isSingleLabel) {
            logits = this.model.predict(idx).squeeze()
        } else {
            const blockSize = this.model.inputs[0].shape[1]
            const idxCond =
                idx.shape[1] <= blockSize
                    ? idx
                    : idx.slice([0, -blockSize], [-1, -1])
            logits = this.model.predict(idxCond)
        }

        if (logits.shape.length === 3) {
            const logitsTimesteps = logits.shape[1]
            const idxTimesteps = idx.shape[1]

            if (logitsTimesteps === idxTimesteps) {
                // If logits and idx have matching timestep lengths
                logits = logits
                    .slice([0, idxTimesteps - 1, 0], [1, 1, logits.shape[2]])
                    .reshape([logits.shape[2]])
            } else {
                // If logits and idx have disparate timestep lengths
                logits = logits
                    .slice([0, logitsTimesteps - 1, 0], [1, 1, logits.shape[2]])
                    .reshape([logits.shape[2]])
            }
        }

        const idxNext = processLogits(logits, idx, {
            doSample,
            temperature,
            topK,
            topP,
            repetitionPenalty
        })

        return idxNext
    })
}

function processLogits(
    logits,
    idx,
    {
        doSample = true,
        temperature = 1.0,
        topK = 0,
        topP = 1.0,
        repetitionPenalty = 1.0
    } = {}
) {
    return tf.tidy(() => {
        let processedLogits = logits

        // Apply repetition penalty if needed
        if (repetitionPenalty !== 1) {
            processedLogits = applyRepetitionPenalty(
                processedLogits,
                idx,
                repetitionPenalty
            )
        }

        // Apply temperature scaling
        if (temperature !== 1) {
            processedLogits = applyTemperature(processedLogits, temperature)
        }

        // Apply top-K filtering
        if (topK > 0) {
            processedLogits = applyTopK(processedLogits, topK)
        }

        // Apply top-P (nucleus) filtering
        if (topP < 1) {
            processedLogits = applyTopP(processedLogits, topP)
        }

        // Apply softmax to convert logits to probabilities
        // const probabilities = tf.softmax(processedLogits)

        if (doSample) {
            return multinomialSampling(processedLogits)
        } else {
            return greedySampling(processedLogits)
        }
    })
}

// this code doesn't really work the way it should
function predictMany({ prompt } = {}) {
    let tokenIndices = this.tokenizer.encode(prompt)

    const inputs = this.tf.tensor2d(
        tokenIndices,
        [1, tokenIndices.length],
        'int32'
    )

    // Predict with the model
    const prediction = this.model.predict(inputs)

    // Squeeze to remove batch dimension since batch size is 1
    const squeezedPred = prediction.squeeze()

    let predictedSequence = []
    for (let i = 0; i < squeezedPred.shape[0]; i++) {
        const timestepPred = squeezedPred.slice([i, 0], [1, -1])

        let sampledIndex = greedySampling(timestepPred)

        predictedSequence.push(sampledIndex.dataSync()[0])
    }

    return this.tokenizer.decode(predictedSequence)
}

function applyTemperature(logits, temperature) {
    return tf.tidy(() => {
        return tf.div(logits, tf.scalar(Math.max(temperature, 1e-6)))
    })
}

function applyTopK(logits, k) {
    return tf.tidy(() => {
        const topK = tf.topk(logits, k)
        const topKIndices = topK.indices
        const topKMask = tf.oneHot(topKIndices, logits.shape[0]).sum(0)
        const maskedLogits = tf.mul(logits, topKMask)
        return maskedLogits
    })
}

function applyTopP(logits, p) {
    return tf.tidy(() => {
        const logitsShape = logits.shape
        const logitsFlat = logits.reshape([-1])
        const topKIndices = tf.topk(logitsFlat, logitsFlat.shape[0]).indices
        const topKLogits = tf.gather(logitsFlat, topKIndices)
        const cumulativeLogits = topKLogits.cumsum()
        const cutoffIndex = cumulativeLogits
            .greater(tf.scalar(p))
            .argMax()
            .flatten()
        const topPIndices = topKIndices.slice(
            [0],
            [cutoffIndex.dataSync()[0] + 1]
        )
        const topPMask = tf.zerosLike(logitsFlat)
        const scatterIndices = tf
            .range(0, topPIndices.shape[0], 1, 'int32')
            .reshape([-1, 1])
        const updateValues = tf.gather(logitsFlat, topPIndices) // Fix: Use the original logits values
        const updatedMask = tf.scatterND(
            scatterIndices,
            updateValues,
            topPMask.shape
        )
        const maskedLogits = updatedMask.reshape(logitsShape)
        return maskedLogits
    })
}

function multinomialSampling(logits) {
    return tf.tidy(() => {
        const sampledIndex = tf.multinomial(logits, 1).reshape([-1])
        return sampledIndex
    })
}

function greedySampling(logits) {
    return tf.tidy(() => {
        return tf.argMax(logits).reshape([-1])
    })
}

function prepareInputs(inputs) {
    return tf.tidy(() => {
        // Check if idx is a tensor or an array
        if (inputs instanceof tf.Tensor) {
            inputs = inputs.clone()
        } else {
            inputs = tf.tensor(inputs)
        }
        // Check data type
        if (inputs.dtype !== 'int32') {
            inputs = inputs.toInt()
        }
        // If the shape of idx is 1D, we need to add a dimension
        if (inputs.shape.length === 1) {
            inputs = inputs.expandDims(0)
        }
        return inputs
    })
}

function applyRepetitionPenalty(logits, outputSequence, repetitionPenalty) {
    return tf.tidy(() => {
        const sequenceLength = outputSequence.shape[1]
        const vocabularySize = logits.shape[0]

        // Create a tensor of shape [vocabularySize] filled with the repetition penalty value
        const penaltyTensor = tf.fill([vocabularySize], repetitionPenalty)

        // Create a mask tensor to identify the previous tokens
        const outputSequenceMask = tf.cast(
            tf.greaterEqual(outputSequence, 0),
            'float32'
        )

        // Create a tensor of shape [sequenceLength] representing the penalty factors
        // The penalty factors decrease linearly from 1 to 0 over the sequence length
        const penaltyFactors = tf.linspace(1, 0, sequenceLength)

        // Calculate the effective penalty for each token in the vocabulary
        const effectivePenalty = tf.tidy(() => {
            const oneHot = tf.oneHot(outputSequence.flatten(), vocabularySize)
            const weightedOneHot = tf.mul(oneHot, penaltyFactors.expandDims(1))
            return weightedOneHot.sum(0)
        })

        // Apply the repetition penalty to the logits
        const penaltyMask = tf.mul(penaltyTensor.sub(1), effectivePenalty)
        const penalizedLogits = logits.sub(penaltyMask)

        return penalizedLogits
    })
}

function enableGradientCheckpointing(model) {
    model.layers.forEach((layer) => {
        if (layer.getClassName() !== 'InputLayer') {
            const originalCall = layer.call

            layer.call = function (inputs, kwargs) {
                const self = this

                return tf.customGrad((x, save) => {
                    const output = originalCall.apply(self, [x, kwargs])
                    save([x])

                    const gradFunc = (dy, saved) => {
                        const inputGrad = tf.mul(dy, saved[0])

                        return [inputGrad]
                    }

                    return { value: output, gradFunc }
                })(inputs[0])
            }
        }
    })

    return model
}

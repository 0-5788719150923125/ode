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
import { trainModel } from './train.js'
import { preprocessData } from './utils.js'

let tf = tfjs

/**
 * The base model class, which defines a standard template
 * that must remain compatible across all model versions.
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
            samplers: customSamplers
        }
        this.model
        this.config = config
        this.tokenizer
    }

    async preInit() {
        this.tf = tfjs
        if (this.config.backend === 'tensorflow') {
            let x = '@tensorflow/tfjs-node-gpu'
            tf = await import(x)
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
        this.defineBuild()
        this.defineOptimizers()
        this.defineSchedulers()
        this.compile()
        this.postInit()
    }

    async load(type = 'file', path = `data/models/ode/model.json`) {
        await this.preInit()
        this.model = await this.tf.loadLayersModel(`${type}://${path}`, {
            strict: true
        })
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
            allocated: allocated.toFixed(2),
            tensors,
            step: this.step,
            batch: this.batch,
            loss: this.loss
        }
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.BasicSubwordTokenizer(
            config?.vocabSize || 6666,
            config?.numIterations || 30_000_000
        )
    }

    defineLossFunctions() {
        this.lossFunctions = [
            {
                function: this.tf.losses.softmaxCrossEntropy,
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
        this.optimizers = [
            this.tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decay || 0.9,
                this.config.momentum || 0.01,
                this.config.epsilon || 1e-8,
                this.config.centered || false
            )
        ]
    }

    defineSchedulers() {
        const learningRate = 0.00333
        // this.optimizers[0].learningRate = learningRate
        this.schedulers = [this.ode.schedulers.constantScheduler(learningRate)]
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
        console.log(`Loaded model: v${this.config.version}`)
        console.log(
            `Tokenizer is ${this.tokenizer.getLength()} tokens in length.`
        )
        console.log(this.config)
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
        return await generateText.call(this, {
            prompt,
            doSample,
            temperature,
            topK,
            topP,
            repetitionPenalty,
            maxNewTokens,
            stopToken
        })
    }

    async train(dataGenerator, args, callbacks) {
        return await trainModel.call(this, dataGenerator, args, callbacks)
    }

    debug(inputs) {
        console.log(new this.ode.layers.DebugLayer().apply(inputs))
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
    const fixedLength = this.config.contextLength
    const isSingleLabel = this.model.outputs[0].shape.length === 2

    return this.tf.tidy(() => {
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
    if (string.endsWith(token)) {
        return true
    }
    return false
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
        let logits, idxNext
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
            logits = logits
                .slice([0, idx.shape[1] - 1, 0], [1, 1, logits.shape[2]])
                .reshape([logits.shape[2]])
        }

        if (repetitionPenalty !== 1) {
            logits = applyRepetitionPenalty(logits, idx, repetitionPenalty)
        }

        if (doSample) {
            if (temperature !== 1) {
                logits = applyTemperature(logits, temperature)
            }
            if (topK > 0) {
                logits = applyTopK(logits, topK)
            }
            if (topP < 1) {
                logits = applyTopP(logits, topP)
            }
            idxNext = sampleFromLogits(logits)
        } else {
            idxNext = greedySampling(logits)
        }

        return idxNext
    })
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

function sampleFromLogits(logits) {
    return tf.tidy(() => {
        const sampledIndex = tf.multinomial(logits, 1).reshape([-1])
        return sampledIndex
    })
}

function greedySampling(logits) {
    return tf.tidy(() => {
        const predictedIndex = tf.argMax(logits)
        return predictedIndex.reshape([-1])
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

        // Create a tensor of shape [sequenceLength, vocabularySize] filled with the repetition penalty value
        const penaltyTensor = tf.fill(
            [sequenceLength, vocabularySize],
            repetitionPenalty
        )

        // Create a mask tensor to identify the previous tokens
        const outputSequenceMask = tf.cast(
            tf.greaterEqual(outputSequence, 0),
            'float32'
        )

        // Reshape the output sequence mask to match the shape of the penalty tensor
        const outputSequenceMaskReshaped = outputSequenceMask
            .reshape([sequenceLength, 1])
            .tile([1, vocabularySize])

        // Create a tensor of shape [sequenceLength, vocabularySize] representing the penalty factors
        // The penalty factors decrease linearly from 1 to 0 over the sequence length
        const penaltyFactors = tf
            .linspace(1, 0, sequenceLength)
            .expandDims(1)
            .tile([1, vocabularySize])

        // Calculate the effective penalty tensor by element-wise multiplication
        const effectivePenalty = tf.mul(
            penaltyTensor,
            tf.mul(outputSequenceMaskReshaped, penaltyFactors)
        )

        // Reshape the effective penalty tensor to match the shape of the logits
        const effectivePenaltyReshaped = effectivePenalty.sum(0).expandDims(0)

        // Apply the repetition penalty to the logits
        const penalizedLogits = tf.sub(logits, effectivePenaltyReshaped)

        return penalizedLogits
    })
}

// function enableGradientCheckpointing(model) {
//     model.layers.forEach((layer) => {
//         const originalCall = layer.call
//         layer.call = function (inputs, kwargs) {
//             const inputTensors = Array.isArray(inputs) ? inputs : [inputs]

//             const output = tf.customGrad((inputs, save) => {
//                 inputs = Array.isArray(inputs) ? inputs : [inputs]

//                 save(inputs)

//                 const output = originalCall.apply(layer, [inputs, kwargs])

//                 save([output])

//                 return {
//                     value: output,
//                     gradFunc: (dy, saved) => {
//                         const savedTensors = saved[0]

//                         const inputs = savedTensors.slice(0, inputs.length)

//                         const gradients = tf.grads((outputs) => outputs[0])(
//                             inputs,
//                             dy
//                         )

//                         return gradients
//                     }
//                 }
//             })(...inputTensors)

//             return output
//         }
//     })

//     return model
// }

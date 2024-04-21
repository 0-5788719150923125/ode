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
import customLayers from './layers.js'
import customLosses from './losses.js'
import customOptimizers from './optimizers.js'
import customTokenizers from './tokenizers.js'
import customSchedulers from './schedulers.js'
import customSamplers from './samplers.js'
import { startTraining } from './train.js'
import { preprocessData } from './utils.js'

/**
 * The base model class, which defines a standard that
 * must remain compatible across all model versions.
 * @constructor
 * @param {Object} config - The configuration settings for the model.
 */
export default class ModelBase {
    constructor(config) {
        this.tf = tf
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

    async init({ corpus = null } = {}) {
        if (corpus) this.config.corpus = corpus
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        this.defineTokenizer()
        if (typeof this.tokenizer.init === 'function') {
            await this.tokenizer.init()
        }
        this.defineLossFunctions()
        this.defineBuild()
        this.defineOptimizers()
        this.defineSchedulers()
        this.compile()
        this.postInit()
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
                function: tf.losses.softmaxCrossEntropy,
                weights: null,
                smoothing: null,
                reduction: tf.Reduction.MEAN
            }
        ]
    }

    defineBuild() {
        throw 'Your model is missing a defineBuild() method. Did you forget to define one?'
    }

    defineOptimizers() {
        this.optimizers = [
            tf.train.rmsprop(
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
        this.optimizers[0].learningRate = learningRate
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
        console.log('Backend:', tf.backend())
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
        maxNewTokens = 50
    } = {}) {
        return await generateText.call(this, {
            prompt,
            doSample,
            temperature,
            topK,
            topP,
            repetitionPenalty,
            maxNewTokens
        })
    }

    async train(dataGenerator, args, callbacks) {
        return await startTraining.call(this, dataGenerator, args, callbacks)
    }

    async save(path = `data/models/ode`) {
        const fs = await import('fs')
        fs.mkdirSync(path, { recursive: true })
        await this.model.save(`file://${path}`, { includeOptimizer: true })
    }

    async load(path = `data/models/ode`) {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        this.defineTokenizer()
        if (typeof this.tokenizer.init === 'function') {
            await this.tokenizer.init()
        }
        this.defineLossFunctions()
        this.model = await tf.loadLayersModel(`file://${path}/model.json`, {
            strict: true
        })
        console.log('successfully loaded model from disk')
        this.defineOptimizers()
        this.defineSchedulers()
        this.compile()
        this.postInit()
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
    maxNewTokens
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

        let decodedText
        for (let step = 0; step < maxNewTokens; step++) {
            const idxNext = predictOnce.call(
                this,
                inputs,
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

function applyRepetitionPenalty(logits, idx, repetitionPenalty) {
    return tf.tidy(() => {
        const logitsShape = logits.shape
        const logitsFlat = logits.reshape([-1])
        const idxFlat = idx.flatten()

        const uniqueIndices = tf.unique(idxFlat).values
        const gatherIndices = uniqueIndices
        const scores = tf.gather(logitsFlat, gatherIndices)

        const penalizedScores = tf.where(
            scores.less(0),
            scores.mul(repetitionPenalty),
            scores.div(repetitionPenalty)
        )

        const scatterIndices = tf
            .range(0, uniqueIndices.shape[0], 1, 'int32')
            .reshape([-1, 1])
        const updatedLogitsFlat = tf.scatterND(
            scatterIndices,
            penalizedScores,
            logitsFlat.shape
        )

        const updatedLogits = updatedLogitsFlat.reshape(logitsShape)

        return updatedLogits
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

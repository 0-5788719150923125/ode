import * as tfjs from '@tensorflow/tfjs'
let tf = tfjs
let isBrowser = true
;(async function () {
    if (typeof window === 'undefined') {
        isBrowser = false
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
 * The base model class, which provides a structure that
 * must remain compatible across all future model versions.
 * @constructor
 * @param {Object} config - The configuration settings for the model.
 */
export default class ModelBase {
    constructor(config) {
        this.tf = tf
        this.isBrowser = isBrowser
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

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        await this.defineTokenizer()
        this.defineLossFunctions()
        this.defineBuild()
        this.defineOptimizers()
        this.defineSchedulers()
        this.compile()
        this.postInit()
    }

    async defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.BasicSubwordTokenizer(
            config?.vocabSize || 6666,
            config?.numIterations || 100_000_000
        )
    }

    defineLossFunctions() {
        this.lossFunctions = [tf.losses.softmaxCrossEntropy]
    }

    defineBuild() {
        throw 'Your model is missing a build() method. Did you forget to define it?'
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
        this.optimizers[0].learningRate = initialLr
        this.schedulers = [this.ode.schedulers.constantScheduler(learningRate)]
    }

    compile() {
        this.model.compile({
            optimizer: this.optimizers[0],
            loss: this.lossFunctions
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

    async generate(prompt, temperature = 0.7, length = 20) {
        if (this.config.mode === 'oneLabel') {
            return await oldGenerateText.call(this, prompt, temperature, length)
        } else return await generateText.call(this, prompt, temperature, length)
    }

    async train(dataGenerator, args) {
        return await startTraining.call(this, dataGenerator, args)
    }

    async save(path = `data/models/ode`) {
        const fs = await import('fs')
        fs.mkdirSync(path, { recursive: true })
        await this.model.save(`file://${path}`, { includeOptimizer: false })
    }

    async load(path = `data/models/ode`) {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        this.defineTokenizer()
        this.defineLossFunctions()
        this.model = await tf.loadLayersModel(`file://${path}/model.json`, {
            // We disable strict mode to prevent errors like this:
            //   ValueError: Provided weight data has no target variable: syn-xxo/w1-qsA
            // TODO: fix the actual problem
            strict: false
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

async function generateText(prompt, temperature, maxNewTokens) {
    let inputs = await prepareInputs.call(this, this.tokenizer.encode(prompt))
    this.tf.tidy(() => {
        for (let step = 0; step < maxNewTokens; step++) {
            const idxNext = generateOnce.call(this, inputs, temperature)
            // Ensure idxNext has a shape of [1, 1] to match the rank of inputs
            const idxNextExpanded = idxNext.expandDims(1) // Adjusting idxNext shape for concatenation
            const idxNew = this.tf.concat([inputs, idxNextExpanded], 1) // Adjusting the axis to 1 for correct concatenation
            this.tf.dispose([inputs, idxNext])
            inputs = this.tf.keep(idxNew)
        }
    })
    const idxArr = await inputs.array()
    this.tf.dispose(inputs)
    return this.tokenizer.decode(idxArr[0])
}

async function oldGenerateText(prompt, temperature = 0.7, maxNewChars = 20) {
    const fixedLength = this.config.contextLength

    // Assuming preprocessData returns an array of token indices
    let tokenIndices = preprocessData(
        prompt,
        this.tokenizer,
        fixedLength,
        'left'
    )

    const generated = tf.tidy(() => {
        let generatedText = prompt

        // Initialize a TensorBuffer for more efficient manipulation
        const inputBuffer = tf.buffer([1, fixedLength], 'int32')

        // Correctly set initial token indices into the buffer
        tokenIndices.forEach((tokenIndex, index) => {
            inputBuffer.set(tokenIndex, 0, index)
        })

        for (let i = 0; i < maxNewChars; i++) {
            // Convert the buffer to a tensor for prediction
            const inputs = inputBuffer.toTensor()

            const output = this.model.predict(inputs).squeeze()

            let winnerIndex
            if (temperature === 0) {
                winnerIndex = greedySampling(output).dataSync()[0]
            } else {
                winnerIndex = temperatureSampling(
                    output,
                    temperature
                ).dataSync()[0]
            }

            if (winnerIndex < 0 || winnerIndex >= this.tokenizer.getLength()) {
                winnerIndex = 0 // Fallback to the first index if out of bounds
            }

            const nextToken = this.tokenizer.decode([winnerIndex])
            generatedText += nextToken

            // Shift left by one position and push the new winnerIndex at the end
            for (let j = 0; j < fixedLength - 1; j++) {
                inputBuffer.set(inputBuffer.get(0, j + 1), 0, j)
            }
            inputBuffer.set(winnerIndex, 0, fixedLength - 1)
        }

        return generatedText
    })

    return generated
}

function generateOnce(idx, temperature) {
    let idxNext
    tf.tidy(() => {
        const block_size = this.model.inputs[0].shape[1]
        const idxCond =
            idx.shape[1] <= block_size
                ? idx
                : idx.slice([0, -block_size], [-1, -1])
        // Forward the model to get the logits for the index in the sequence
        const logits = this.model.predict(idxCond)

        let logitsScaled
        if (logits.shape.length === 3) {
            // pluck the logits at the final step for timeDistributed
            logitsScaled = logits
                .slice([0, idx.shape[1] - 1, 0], [1, 1, logits.shape[2]])
                .reshape([logits.shape[2]])
        } else {
            // For oneLabel mode, logits is already in the expected shape
            logitsScaled = logits
        }

        // either sample from a scaled distribution or take the most likely element
        if (temperature !== 1) {
            idxNext = temperatureSampling(logitsScaled, temperature)
        } else {
            idxNext = greedySampling(logitsScaled)
        }

        tf.keep(idxNext)
    })
    return idxNext
}

function greedySampling(probabilities) {
    const index = tf.tidy(() => {
        const predictedIndex = tf.argMax(probabilities)
        return predictedIndex.reshape([-1])
    })
    return index
}

function temperatureSampling(logits, temperature) {
    return tf.tidy(() => {
        const probabilities = tf.div(
            logits,
            tf.scalar(Math.max(temperature, 1e-6))
        )
        const sampledIndex = tf.multinomial(probabilities, 1).reshape([-1])
        return sampledIndex
    })
}

function prepareInputs(inputs) {
    tf.tidy(() => {
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
        tf.keep(inputs)
        // keep idx from deletion
    })
    return inputs
}

function topKSampling(logits, k) {
    return tf.tidy(() => {
        // Step 1: Use tf.topk to find the top k logits and their indices
        const topk = tf.topk(logits, k)
        const values = topk.values
        const indices = topk.indices

        // Step 2: Calculate the probabilities of the top k logits
        // Normalize the values to prevent large logits from dominating
        // const scaledValues = values.sub(values.max()).softmax()
        const scaledValues = values.sub(values.max())

        // Step 3: Sample from the top k values
        const sampledIndex = tf
            .multinomial(scaledValues, 1, null, false)
            .dataSync()[0]

        // Step 4: Map the sampled index back to the original logits' space using the indices tensor
        const originalIndex = indices.arraySync()[sampledIndex]

        return originalIndex
    })
}

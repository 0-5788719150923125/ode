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
import { BasicSubwordTokenizer } from './tokenizers.js'
import { DebugLayer } from './layers.js'
import { startTraining } from './train.js'
import { preprocessData, stringSampler } from './utils.js'

/**
 * The base model class, which represents a syntax and structure that must
 * be remain compatible across all model versions.
 * @constructor
 * @param {Object} config - The configuration settings for the model.
 */
export default class ModelBase {
    constructor(config) {
        this.tf = tf
        this.model
        this.config = config
        this.tokenizer
    }

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')

        tf.enableProdMode()

        console.log('Backend:', tf.backend())
        console.log(this.config)

        this.setupTokenizer()
        await this.tokenizer.writeVocabularyToFile()

        if (this.config.loadFromFile) {
            console.log('loading from file')
            await this.load()
        } else {
            this.build()
        }
        await this.compile()
        this.postInit()
    }

    setupTokenizer() {
        this.tokenizer = new BasicSubwordTokenizer(6666, 250_000_000)
    }

    build() {
        // pass
    }

    async compile() {
        // Compile the model
        this.lossFunctions = [tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decay || 0.9,
                this.config.momentum || 0.01,
                this.config.epsilon || 1e-8,
                false
            ),
            loss: this.lossFunctions
        })
    }

    postInit() {
        console.log(this.model.optimizer)
        console.log(this.model.summary())
        console.log(`Loaded model: v${this.config.version}`)
        console.log(
            `Tokenizer is ${this.tokenizer.getLength()} tokens in length.`
        )
    }

    async generate(seed, temperature = 0.7, length = 20, greedy) {
        return await generateText.call(this, seed, temperature, length)
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
        this.model = await tf.loadLayersModel(`file://${path}/model.json`)
        console.log('successfully loaded model from disk')
    }

    debug(inputs) {
        const layer = new DebugLayer()
        console.log(layer.apply(inputs))
    }

    sampler(type = 'string') {
        return stringSampler
    }
}

async function generateText(prompt, temperature = 0.7, maxNewChars = 20) {
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
                winnerIndex = greedySampling(output)
            } else {
                winnerIndex = temperatureSampling(output, temperature)
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

export function greedySampling(probabilities) {
    const index = tf.tidy(() => {
        const predictedIndex = tf.argMax(probabilities)
        return predictedIndex.dataSync()[0]
    })
    return index
}

export function temperatureSampling(logits, temperature) {
    return tf.tidy(() => {
        const probabilities = tf.div(
            tf.softmax(logits),
            Math.max(temperature, 1e-6)
        )
        const sampledIndex = tf.multinomial(probabilities, 1).dataSync()[0]
        return sampledIndex
    })
}

export function topKSampling(logits, k) {
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

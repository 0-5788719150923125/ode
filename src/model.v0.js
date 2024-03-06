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
import { startTraining } from './train.js'
import { preprocessData } from './utils.js'

export default class ModelBase {
    constructor(config) {
        this.model
        this.config = config
        this.padToken = '�'
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
            )
        )
        this.vocab.unshift(this.padToken)
    }

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        tf.enableProdMode()
        console.log('Backend:', tf.backend())
        this.build()
        this.postInit()
    }

    build() {
        // pass
    }

    postInit() {
        console.log(this.model.summary())
        console.log(this.model.optimizer)
    }

    async generate(seed, temperature = 0.7, length = 20, greedy = false) {
        return await generateText.call(this, seed, temperature, length, greedy)
    }

    async train(dataGenerator, args) {
        return await startTraining.call(this, dataGenerator, args)
    }

    async save(path = `data/models/ode`) {
        const fs = await import('fs')
        fs.mkdirSync(path, { recursive: true })
        await this.model.save(`file://${path}`, { includeOptimizer: false })
    }
}

async function generateText(prompt, temperature = 0.7, maxNewChars = 20) {
    const fixedLength = this.config.maxSequenceLength

    const generated = tf.tidy(() => {
        let generated = prompt

        let tokenIndices = preprocessData(
            prompt,
            this.vocab,
            fixedLength,
            'left'
        )

        let inputs = tf.tensor2d([tokenIndices], [1, fixedLength], 'int32')

        for (let i = 0; i < maxNewChars; i++) {
            const output = this.model.predict(inputs).squeeze()

            let winnerIndex
            if (temperature === 0) {
                winnerIndex = greedySampling(output)
            } else {
                // winnerIndex = topKSampling(output, 4)
                winnerIndex = temperatureSampling(output, temperature)
            }

            if (winnerIndex < 0 || winnerIndex >= this.vocab.length) {
                winnerIndex = 0 // Fallback to the first index if out of bounds
            }

            const nextChar = this.vocab[winnerIndex]
            generated += nextChar

            // Update tokenIndices and inputTensor for the next iteration
            tokenIndices.push(winnerIndex)
            if (tokenIndices.length > fixedLength) {
                tokenIndices.shift() // Remove the oldest token
            }

            inputs = tf.tensor2d([tokenIndices], [1, fixedLength], 'int32')
        }
        return generated
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
        const probabilities = tf.div(logits, Math.max(temperature, 1e-6))
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

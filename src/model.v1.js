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

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.padToken = '�'
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
            )
        )
        this.vocab.unshift(this.padToken)
        this.model = tf.sequential()
    }

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')

        tf.enableProdMode()

        console.log('Backend:', tf.backend())

        // Add the embedding layer as the first layer
        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length, // Should match size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                embeddingsInitializer: 'glorotUniform',
                embeddingsConstraint: tf.constraints.maxNorm({
                    maxValue: 0.1
                }),
                embeddingsRegularizer: tf.regularizers.l2(),
                maskZero: true
            })
        )

        // Apply dropout on the embeddings layer
        this.model.add(tf.layers.dropout({ rate: 0.1 }))

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        dropout: 0.1,
                        stateful: false,
                        activation: 'softsign',
                        kernelInitializer: 'glorotUniform',
                        kernelConstraint: tf.constraints.maxNorm({
                            axis: 0,
                            maxValue: 2.0
                        }),
                        recurrentActivation: 'sigmoid',
                        recurrentInitializer: 'orthogonal',
                        recurrentConstraint: tf.constraints.maxNorm({
                            axis: 0,
                            maxValue: 2.0
                        }),
                        returnSequences: i < this.config.layout.length - 1 // False for the last GRU layer
                    }),
                    mergeMode: 'concat'
                })
            )
            this.model.add(
                tf.layers.layerNormalization({
                    epsilon: 1e-3
                })
            )
        })

        // this.model.add(
        //     tf.layers.dense({
        //         units: 64,
        //         activation: 'swish'
        //     })
        // )

        // Add the final dense layer with softmax activation
        this.model.add(
            tf.layers.dense({
                units: this.vocab.length,
                activation: 'linear'
            })
        )

        // Compile the model
        this.model.compile({
            optimizer: tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decay || 0,
                this.config.momentum || 0,
                this.config.epsilon || 1e-8
            ),
            loss: [tf.losses.softmaxCrossEntropy]
        })

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

            // Efficiently update the input tensor by shifting it and appending the new token
            tf.dispose(inputs)
            inputs = tf.tensor2d([tokenIndices], [1, fixedLength], 'int32')

            tf.dispose(output)
        }

        tf.dispose(inputs)

        return generated
    })

    return generated
}

function greedySampling(probabilities) {
    const index = tf.tidy(() => {
        const predictedIndex = tf.argMax(probabilities)
        return predictedIndex.dataSync()[0]
    })
    return index
}

function temperatureSampling(logits, temperature) {
    return tf.tidy(() => {
        const scaled = logits.div(tf.scalar(temperature))
        const probabilities = scaled.softmax()
        const sampledIndex = tf.multinomial(probabilities, 1).dataSync()[0]
        return sampledIndex
    })
}

function multinomialSampling(logits, temperature) {
    const probabilities = tf.div(tf.log(logits), Math.max(temperature, 1e-6))
    const scaledProbs = tf.softmax(probabilities)
    const predictions = tf.multinomial(scaledProbs, 1)
    const index = predictions.dataSync()[0]
    tf.dispose([probabilities, predictions])
    return index
}

function stochasticSampling(probabilities) {
    // Convert probabilities to a flat array
    const probsArray = probabilities.dataSync()
    // Calculate the cumulative sum of the probabilities
    const cumulativeSum = probsArray.map(
        (
            (sum) => (value) =>
                (sum += value)
        )(0)
    )
    // Generate a random number between 0 and 1
    const random = Math.random() * cumulativeSum[cumulativeSum.length - 1]
    // Find the first index where the cumulative sum is greater than the random number
    const selectedIndex = cumulativeSum.findIndex((sum) => sum > random)
    return selectedIndex
}

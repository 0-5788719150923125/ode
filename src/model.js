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

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.vocab = Array.from(
            new Set(
                `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'";:(){}[]<>#~-_|/\\\n `
            )
        )
        this.padToken = '¶'
        this.vocab.unshift(this.padToken)
        this.model = tf.sequential()
    }

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 256000000)

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
        this.model.add(tf.layers.dropout({ rate: 0.1 }))

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        kernelInitializer: 'glorotUniform',
                        kernelConstraint: tf.constraints.maxNorm({ axis: 0 }),
                        activation: 'swish',
                        recurrentInitializer: 'orthogonal',
                        recurrentActivation: 'sigmoid',
                        recurrentConstraint: tf.constraints.maxNorm({
                            axis: 0
                        }),
                        dropout: 0.1,
                        returnSequences: i < this.config.layout.length - 1 // Set to false for the last GRU layer
                    }),
                    mergeMode: 'ave'
                })
            )
            this.model.add(
                tf.layers.layerNormalization({
                    epsilon: 1e-5
                })
            )
        })

        // Add the final dense layer with softmax activation
        this.model.add(
            tf.layers.dense({
                units: this.vocab.length,
                activation: 'softmax'
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
            loss: [tf.metrics.categoricalCrossentropy]
        })

        console.log(this.model.summary())
        console.log(this.model.optimizer)
    }

    async generate(seed, temperature = 0.7, length = 20) {
        return await generate.call(this, seed, temperature, length)
    }

    async train(dataGenerator, args) {
        return await startTraining.call(this, dataGenerator, args)
    }

    async save(path = `file://models/ode`) {
        await this.model.save(path, { includeOptimizer: false })
    }
}

async function generate(prompt, temperature = 0.7, maxLength = 20) {
    let tokenIndices = Array.from(prompt).map((e) => this.vocab.indexOf(e))

    const fixedLength = this.config.maxSequenceLength

    if (tokenIndices.length > fixedLength) {
        tokenIndices = tokenIndices.slice(tokenIndices.length - fixedLength)
    } else if (tokenIndices.length < fixedLength) {
        tokenIndices = new Array(fixedLength - tokenIndices.length)
            .fill(0)
            .concat(tokenIndices)
    }

    let generated = prompt

    for (let i = 0; i < maxLength; i++) {
        const input = tf.tensor2d([tokenIndices], [1, fixedLength], 'int32')

        const output = this.model.predict(input).squeeze()

        let winnerIndex = await sample(output, temperature)

        if (winnerIndex < 0 || winnerIndex >= this.vocab.length) {
            winnerIndex = 0 // Fallback to the first index if out of bounds
        }

        const nextChar = this.vocab[winnerIndex]
        generated += nextChar

        tokenIndices.push(winnerIndex)
        if (tokenIndices.length > fixedLength) {
            tokenIndices.shift()
        }

        tf.dispose([input, output])
    }

    return generated
}

async function sample(probabilities, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(
            tf.log(probabilities),
            Math.max(temperature, 1e-6)
        )
        const normalized = false
        return tf.multinomial(logits, 1, null, normalized).dataSync()[0]
    })
}

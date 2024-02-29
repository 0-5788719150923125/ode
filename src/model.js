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
import { trainModel } from './train.js'

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\\n `
            )
        )
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
                inputDim: this.vocab.length, // Size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                embeddingsInitializer: 'glorotUniform',
                embeddingsConstraint: tf.constraints.maxNorm({
                    maxValue: 0.05
                }),
                embeddingsRegularizer: 'l1l2'
            })
        )

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        kernelInitializer: 'glorotUniform',
                        recurrentInitializer: 'orthogonal',
                        biasInitializer: 'zeros',
                        kernelConstraint: tf.constraints.maxNorm({ axis: 0 }),
                        recurrentConstraint: tf.constraints.maxNorm({
                            axis: 0
                        }),
                        dropout: 0.2,
                        returnSequences: i < this.config.layout.length - 1 // Set to false for the last GRU layer
                    }),
                    mergeMode: 'concat'
                })
            )
        })

        // Add the final Dense layer with softmax activation
        this.model.add(
            tf.layers.dense({
                units: this.vocab.length,
                activation: 'softmax'
            })
        )

        // Compile the model
        this.lossFunction = tf.metrics.categoricalCrossentropy
        this.model.compile({
            optimizer: tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decayRate || 0,
                this.config.momentum || 0,
                this.config.epsilon || 1e-8
            ),
            loss: this.lossFunction
        })
    }

    async generate(seed, temperature = 0.7, length = 20) {
        const bound = generate.bind(this)
        return await bound(seed, temperature, length)
    }

    async train(
        dataGenerator,
        batchSize,
        gradientAccumulationSteps,
        sampleLen,
        generateEvery
    ) {
        const bound = trainModel.bind(this)
        await bound(
            dataGenerator,
            batchSize,
            gradientAccumulationSteps,
            sampleLen,
            generateEvery
        )
    }

    async save(path = `file://data/model`) {
        await this.model.save(path)
    }
}

async function generate(prompt, temperature = 0.7, maxLength = 20) {
    let generated = prompt
    const tokenIndices = Array.from(prompt).map((e) => this.vocab.indexOf(e))

    for (let i = 0; i < maxLength; i++) {
        const input = tf.tensor2d(
            tokenIndices,
            [1, tokenIndices.length],
            'int32'
        )

        const logits = this.model.predict(input)

        const winnerIndex = await sample(tf.squeeze(logits), temperature)

        if (winnerIndex < 0 || winnerIndex >= this.vocab.length) {
            winnerIndex = 0 // Fallback to the first index if out of bounds
        }

        const nextChar = this.vocab[winnerIndex]
        generated += nextChar
        tokenIndices.push(winnerIndex)

        tf.dispose([logits, input])
    }

    return generated
}

async function sample(probabilities, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(
            tf.log(probabilities),
            Math.max(temperature, 1e-6)
        )
        const isNormalized = false
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0]
    })
}

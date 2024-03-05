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
                // embeddingsInitializer: 'glorotUniform',
                // embeddingsConstraint: tf.constraints.maxNorm({
                //     maxValue: 0.1
                // }),
                // embeddingsRegularizer: tf.regularizers.l2(),
                maskZero: true
            })
        )

        // Apply dropout on the embeddings layer
        // this.model.add(tf.layers.dropout({ rate: 0.1 }))

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        // dropout: 0.1,
                        stateful: false,
                        activation: 'tanh',
                        // kernelInitializer: 'glorotUniform',
                        // kernelConstraint: tf.constraints.maxNorm({
                        //     axis: 0,
                        //     maxValue: 2.0
                        // }),
                        // recurrentActivation: 'sigmoid',
                        // recurrentInitializer: 'orthogonal',
                        // recurrentConstraint: tf.constraints.maxNorm({
                        //     axis: 0,
                        //     maxValue: 2.0
                        // }),
                        returnSequences: i < this.config.layout.length - 1 // False for the last GRU layer
                    }),
                    mergeMode: 'concat'
                })
            )
            // this.model.add(
            //     tf.layers.layerNormalization({
            //         epsilon: 1e-3
            //     })
            // )
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
            loss: ['categoricalCrossentropy']
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
    let generated = prompt

    const fixedLength = this.config.maxSequenceLength

    let tokenIndices = preprocessData(prompt, this.vocab, fixedLength, 'left')

    let inputs = tf.tensor2d([tokenIndices], [1, fixedLength], 'int32')

    for (let i = 0; i < maxNewChars; i++) {
        const output = this.model.predict(inputs).squeeze()

        let winnerIndex = await greedySample(output, temperature)

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
}

async function greedySample(probabilities, temperature) {
    const logits = tf.div(tf.log(probabilities), Math.max(temperature, 1e-6))
    const normalized = false
    const predictions = tf.multinomial(logits, 1, null, normalized)
    const index = await predictions.data().then((data) => data[0])
    tf.dispose([logits, predictions])
    return index
}

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
                        returnSequences: true // False for the last GRU layer
                    }),
                    mergeMode: 'ave'
                })
            )
            this.model.add(
                tf.layers.layerNormalization({
                    epsilon: 1e-3
                })
            )
        })

        // Add the final dense layer with softmax activation
        this.model.add(
            tf.layers.timeDistributed({
                layer: tf.layers.dense({
                    units: this.vocab.length,
                    activation: 'softmax'
                })
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

function preprocessText(texts, vocab, expectedSequenceLength) {
    const inputIndices = texts.map((text) => {
        const chars = text.split('')
        const indices = chars.map((char) => vocab.indexOf(char))

        // Pad on the left
        const padding = new Array(expectedSequenceLength - indices.length).fill(
            0
        )
        const paddedIndices = padding
            .concat(indices)
            .slice(indices.length, expectedSequenceLength)
            .concat(indices)

        return paddedIndices.map((i) => i)
    })

    return inputIndices
}

async function generateText(prompt, temperature = 0.7, maxLength = 20) {
    let generated = prompt
    const maxSequenceLength = this.config.contextLength

    prompt = prompt.slice(-maxSequenceLength)

    // Initialize input sequence data
    let inputSequence = preprocessText([prompt], this.vocab, maxSequenceLength)

    const inputs = tf.tensor2d(inputSequence, [
        inputSequence.length,
        maxSequenceLength
    ])

    const output = this.model.predict([inputs])

    const nextSequence = await sampleSequences.call(this, output, temperature)

    generated += nextSequence

    tf.dispose(output)

    return generated
}

async function sampleSequences(
    probabilities,
    temperature = 0.7,
    greedy = false
) {
    // Reshape the probabilities if needed (assumes output is [batchSize, sequenceLength, vocabSize])
    const reshapedProbabilities = probabilities.reshape([
        probabilities.shape[1],
        probabilities.shape[2]
    ])

    let logits, predictions, predictedIndices
    if (greedy) {
        // Greedy implementation
        predictedIndices = reshapedProbabilities.argMax(-1).dataSync()
    } else {
        // Apply temperature scaling to logits
        logits = tf.div(
            tf.log(reshapedProbabilities),
            Math.max(temperature, 1e-6)
        )

        // Sample using multinomial
        predictions = tf.multinomial(logits, 1, null, false)
        predictedIndices = await predictions.data()
    }

    // Map indices to characters
    const sequence = predictedIndices.map((index) => this.vocab[index])

    tf.dispose(predictions)
    return sequence.join('')
}

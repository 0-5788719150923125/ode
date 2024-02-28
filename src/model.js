import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-webgl'
import { trainModel } from './train.js'

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.vocab = [
            '<pad>',
            ...new Set(
                Array.from(
                    `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\\n `
                )
            )
        ]

        console.log(this.vocab)
        this.model = tf.sequential()
    }

    async init() {
        await tf.setBackend(this.config.backend || 'cpu')
        console.log('Backend:', tf.backend())
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 256000000)

        // Add the embedding layer as the first layer
        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length, // Size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                inputLength: this.config.inputLength // Length of input sequences
            })
        )

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.gru({
                    units: layer,
                    returnSequences: i < this.config.layout.length - 1 // Set to false for the last GRU layer
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
        this.model.compile({
            optimizer: tf.train.rmsprop(this.config.learningRate),
            loss: 'categoricalCrossentropy'
        })
    }

    getModel() {
        return this.model
    }

    async train(dataGenerator, batchSize) {
        const bound = trainModel.bind(this)
        await bound(dataGenerator, batchSize)
    }

    getWeights() {
        return this.model.getWeights()
    }

    async generate(seed, temperature = 0.7, length = 20) {
        const bound = generate.bind(this)
        return await bound(seed, temperature, length)
    }

    async saveModel() {
        // Define a path to save the model
        const savePath = `file://data/model`

        // Save the model
        await this.model.save(savePath)

        console.log(`Model saved to ${savePath}`)
    }
}

async function generate(seed, temperature, maxLength = 20) {
    let sentenceIndices =
        seed.length > 0
            ? Array.from(seed)
                  .map((e) => this.vocab.indexOf(e))
                  .filter((index) => index !== -1)
            : [this.vocab.indexOf('<pad>')] // Start with a single pad token if no seed

    let generated = seed

    for (let i = 0; generated.length < maxLength; i++) {
        // Dynamically create input sequence with minimal padding
        let inputIndices = sentenceIndices.slice(-this.config.inputLength)
        if (inputIndices.length < this.config.inputLength) {
            inputIndices = Array(this.config.inputLength - inputIndices.length)
                .fill(0)
                .concat(inputIndices) // Minimal padding to meet inputLength
        }

        const input = tf.tensor2d(
            [inputIndices],
            [1, this.config.inputLength],
            'int32'
        )

        const logits = this.model
            .predict(input)
            .squeeze()
            .div(tf.scalar(temperature))
        const winnerIndex = tf.multinomial(logits, 1).dataSync()[0]
        logits.dispose()
        input.dispose()

        if (winnerIndex >= 0 && winnerIndex < this.vocab.length) {
            const nextChar = this.vocab[winnerIndex]
            generated += nextChar
            sentenceIndices.push(winnerIndex)

            // Now focus on the generated output, progressively reducing padding influence
            if (sentenceIndices.length > this.config.inputLength) {
                sentenceIndices.shift()
            }
        } else {
            break // Stop if an invalid index is generated
        }
    }

    return generated
}

import * as tf from '@tensorflow/tfjs'
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
                inputDim: this.vocab.length + 1, // Size of the vocabulary
                outputDim: this.config.embeddingDimensions // Dimension of the embedding vectors
                // inputLength: this.config.inputLength // Length of input sequences
            })
        )

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        returnSequences: i < this.config.layout.length - 1 // Set to false for the last GRU layer
                    }), // Each direction of the Bidirectional layer has 'layer' GRU units
                    mergeMode: 'concat' // Decide how to merge forward and backward states
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

async function generate(seed, temperature = 0.7, maxLength = 20) {
    let sentenceIndices = Array.from(seed).map((e) => this.vocab.indexOf(e)) // Guarantee in-vocab seed characters

    let generated = seed

    for (let i = 0; i < maxLength; i++) {
        // Slice out only the necessary input sequence:
        // const inputIndices = sentenceIndices.slice(-this.config.inputLength)
        let inputIndices = sentenceIndices // Use all sentenceIndices

        const input = tf.tensor2d(
            [inputIndices],
            [1, inputIndices.length], // Dynamically set the second dimension of the shape
            'int32'
        )

        const logits = this.model
            .predict(input)
            .squeeze()
            .div(tf.scalar(temperature))

        const winnerIndex = tf.multinomial(logits, 1).dataSync()[0]
        logits.dispose()
        input.dispose()

        if (winnerIndex < 0 || winnerIndex > this.vocab.length) {
            console.warn('Invalid index generated, breaking')
            break // Stop if an invalid index is generated
        }

        const nextChar = this.vocab[winnerIndex]
        generated += nextChar
        sentenceIndices.push(winnerIndex)
    }

    return generated
}

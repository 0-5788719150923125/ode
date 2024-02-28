// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs'
// Add the WebGPU backend to the global backend registry.
// import '@tensorflow/tfjs-backend-webgpu'
// Set the backend to WebGPU and wait for the module to be ready.
// tf.setBackend('webgpu').then(() => main())
import { trainModel } from './train.js'

console.log('Backend:', tf.backend())
tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 256000000)

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.vocab = Array.from(
            new Set(
                Array.from(
                    `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\\n `
                )
            )
        )
        this.model = tf.sequential()
        this.init()
    }

    init() {
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
    let sentenceIndices = Array.from(seed)
        .map((e) => this.vocab.indexOf(e))
        .filter((index) => index !== -1)

    // Initialize generated text with the seed
    let generated = seed

    while (generated.length < maxLength) {
        // Pad the sentenceIndices to ensure it has the required length
        const paddedSentenceIndices = new Array(
            this.config.inputLength - sentenceIndices.length
        )
            .fill(0)
            .concat(sentenceIndices)

        // Prepare the input tensor with the shape [1, inputLength - 1]
        const input = tf.tensor2d(
            [paddedSentenceIndices],
            [1, this.config.inputLength],
            'int32'
        )

        // Predict the next character
        const logits = this.model
            .predict(input)
            .squeeze()
            .div(tf.scalar(temperature))
        const winnerIndex = tf.multinomial(logits, 1).dataSync()[0]
        logits.dispose() // Dispose the logits tensor immediately after use

        if (winnerIndex >= 0 && winnerIndex < this.vocab.length) {
            const nextChar = this.vocab[winnerIndex]
            generated += nextChar
            sentenceIndices.push(winnerIndex) // Append the winner index to the sentenceIndices

            // Keep only the most recent ${inputLength} indices for the next prediction
            if (sentenceIndices.length > this.config.inputLength) {
                sentenceIndices.shift() // Remove the oldest index
            }
        } else {
            break // Break the loop if winnerIndex is invalid
        }

        input.dispose() // Dispose the input tensor after each prediction
    }

    return generated
}

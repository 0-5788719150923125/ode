import * as tf from '@tensorflow/tfjs-node'
import { trainModel } from './train'

console.log('Backend:', tf.backend())

export default class ModelPrototype {
    constructor(lstmLayerSize, sampleLen, learningRate, displayLength) {
        this.lstmLayerSize = lstmLayerSize
        this.sampleLen = sampleLen
        this.vocab = Array.from(
            new Set(
                Array.from(
                    `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\\n `
                )
            )
        )
        this.learningRate = learningRate
        this.displayLength = displayLength
        this.model = null
        this.init()
    }

    init() {
        // Initialize the model as a sequential model
        this.model = tf.sequential()

        // Add the embedding layer as the first layer
        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length, // Size of the vocabulary
                outputDim: 64, // Dimension of the embedding vectors
                inputLength: this.sampleLen - 1 // Length of input sequences
            })
        )

        // Add LSTM layers
        // Adjust the last LSTM layer in the init method
        this.lstmLayerSize.forEach((lstmLayerSize, i) => {
            this.model.add(
                tf.layers.lstm({
                    units: lstmLayerSize,
                    returnSequences: i < this.lstmLayerSize.length - 1 // Set to false for the last LSTM layer
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
            optimizer: tf.train.rmsprop(this.learningRate),
            loss: 'categoricalCrossentropy'
        })
    }

    getModel() {
        return this.model
    }

    async trainModel(dataGenerator) {
        const bound = trainModel.bind(this)
        await bound(dataGenerator)
    }

    getWeights() {
        return this.model.getWeights()
    }

    async generate(seed, temperature = 0.7) {
        const bound = generate.bind(this)
        return await bound(seed, temperature)
    }
}

async function generate(seed, temperature, maxLength = 20) {
    let sentenceIndices = Array.from(seed)
        .map((e) => this.vocab.indexOf(e))
        .filter((index) => index !== -1)

    // Adjust the length of sentenceIndices to match the model's expected input length
    if (sentenceIndices.length > this.sampleLen - 1) {
        sentenceIndices = sentenceIndices.slice(0, this.sampleLen - 1)
    } else {
        while (sentenceIndices.length < this.sampleLen - 1) {
            sentenceIndices.unshift(0) // Prepend with zeros
        }
    }

    let generated = seed // Start with the seed

    while (generated.length < maxLength) {
        const input = tf.tensor2d(
            [sentenceIndices],
            [1, this.sampleLen - 1],
            'int32'
        )
        const output = this.model.predict(input)

        const winnerIndex = tf.tidy(() => {
            // Ensure temperature is a numeric value
            const temp = parseFloat(temperature)
            const logits = output.squeeze().div(tf.scalar(temp))
            return tf.multinomial(logits, 1).dataSync()[0]
        })

        if (winnerIndex >= 0 && winnerIndex < this.vocab.length) {
            const nextChar = this.vocab[winnerIndex]
            generated += nextChar
            sentenceIndices = [...sentenceIndices.slice(1), winnerIndex]
        } else {
            break // Break the loop if winnerIndex is invalid
        }

        input.dispose()
        output.dispose()
    }

    return generated
}

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
        console.log(this.vocab)
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

    if (sentenceIndices.length > this.sampleLen - 1) {
        sentenceIndices = sentenceIndices.slice(0, this.sampleLen - 1)
    } else {
        while (sentenceIndices.length < this.sampleLen - 1) {
            sentenceIndices.unshift(0) // Prepend with zeros
        }
    }

    let generated = seed // Start with the seed
    const tempScalar = tf.scalar(parseFloat(temperature)) // Create the temperature scalar outside the loop

    // Warm-up prediction to initialize the model
    const warmUpInput = tf.tensor2d(
        [sentenceIndices],
        [1, this.sampleLen - 1],
        'int32'
    )
    this.model.predict(warmUpInput).dispose()
    warmUpInput.dispose()

    while (generated.length < maxLength) {
        const input = tf.tensor2d(
            [sentenceIndices],
            [1, this.sampleLen - 1],
            'int32'
        )

        // Predict next character
        const logits = this.model.predict(input).squeeze().div(tempScalar)
        const winnerIndex = tf.multinomial(logits, 1).dataSync()[0]
        logits.dispose() // Dispose the logits tensor immediately after use

        if (winnerIndex >= 0 && winnerIndex < this.vocab.length) {
            const nextChar = this.vocab[winnerIndex]
            generated += nextChar
            sentenceIndices = [...sentenceIndices.slice(1), winnerIndex]
        } else {
            break // Break the loop if winnerIndex is invalid
        }

        input.dispose() // Dispose tensors at the end of each loop iteration
    }

    tempScalar.dispose() // Dispose the temperature scalar after the loop
    return generated
}

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

async function generate(seed, temperature) {
    // XXX: Fetch the sequence of numeric values which correspond to the
    //      sentence.
    let sentenceIndices = Array.from(seed).map((e) => this.vocab.indexOf(e))

    let generated = ''

    // XXX: Note that since the displayLength is arbitrary, we can make it
    //      much larger than our sampleLen. This loop will continue to iterate
    //      about the sentenceIndices and buffering the output of the network,
    //      which permits it to continue generating far past our initial seed
    //      has been provided.
    while (generated.length < this.displayLength) {
        const inputBuffer = new tf.TensorBuffer([
            1,
            this.sampleLen,
            this.vocab.length
        ])

        ;[...Array(this.sampleLen)].map((_, i) =>
            inputBuffer.set(1, 0, i, sentenceIndices[i])
        )

        const input = inputBuffer.toTensor()
        const output = this.model.predict(input)

        // XXX: Pick the character the RNN has decided is the most likely.
        //      tf.tidy cleans all of the allocated tensors within the function
        //      scope after it has been executed.
        const [winnerIndex] = tf.tidy(() =>
            // XXX: Draws samples from a multinomial distribution (these are distributions
            //      involving multiple variables).
            //      tf.squeeze remove dimensions of size (1) from the supplied tensor. These
            //      are then divided by the specified temperature.
            tf
                .multinomial(
                    // XXX: Use the temperature to control the network's spontaneity.
                    tf.div(
                        tf.log(tf.squeeze(output)),
                        Math.max(temperature, 1e-6)
                    ),
                    1,
                    null,
                    false
                )
                .dataSync()
        )

        // XXX: Always clean up tensors once you're finished with them to improve
        //      memory utilization and prevent leaks.
        input.dispose()
        output.dispose()

        // XXX: Here we append the generated character to the resulting string, and
        //      add this char to the sliding window along the sentenceIndices. This
        //      is how we continually wrap around the same buffer and generate arbitrary
        //      sequences of data even though our network only accepts fixed inputs.
        generated += this.vocab[winnerIndex]
        sentenceIndices = sentenceIndices.slice(1)
        sentenceIndices.push(winnerIndex)
    }
    // console.log(`Generated text (temperature=${temperature}):\n ${generated}\n`)
    return generated
}

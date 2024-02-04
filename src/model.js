import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'
import { trainModel } from './train'

export default class ModelPrototype {
    constructor(lstmLayerSize, sampleLen, learningRate, displayLength) {
        this.lstmLayerSize = lstmLayerSize
        this.sampleLen = sampleLen
        this.characters = Array.from(
            new Set(
                Array.from(
                    `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]| `
                )
            )
        )
        this.learningRate = learningRate
        this.displayLength = displayLength
        this.model = null
        this.init()
    }

    init() {
        // XXX: Create our processing model. We iterate through the array of lstmLayerSize and
        //      iteratively add an LSTM processing layer whose number of internal units match the
        //      specified value.
        this.model = this.lstmLayerSize.reduce(
            (mdl, lstmLayerSize, i, orig) => {
                mdl.add(
                    tf.layers.lstm({
                        units: lstmLayerSize,
                        // XXX: For all layers except the last one, we specify that we'll be returning
                        //      sequences of data. This allows us to iteratively chain individual LSTMs
                        //      to one-another.
                        returnSequences: i < orig.length - 1,
                        // XXX: Since each LSTM layer generates a sequence of data, only the first layer
                        //      needs to receive a specific input shape. Here, we initialize the inputShape
                        //      [sampleLen, this.characters.length]. This defines that the first layer will receive an
                        //      input matrix which allows us to convert from our selected sample range into
                        //      the size of our charset. The charset uses one-hot encoding, which allows us
                        //      to represent each possible character in our dataset using a dedicated input
                        //      neuron.
                        inputShape:
                            i === 0
                                ? [this.sampleLen, this.characters.length]
                                : undefined
                    })
                )
                // XXX: Here we use a sequential processing model for our network. This model gets passed
                //      between each iteration, and is what we add our LSTM layers to.
                return mdl
            },
            tf.sequential()
        )

        // XXX: At the output, we use a softmax function (a normalized exponential) as the final
        //      classification layer. This is common in many neural networks. It's particularly
        //      important for this example, because we use the logit probability model (which
        //      supports regression for networks with more than 2 possible outcomes of a categorically
        //      distributed dependent variable).
        this.model.add(
            tf.layers.dense({
                units: this.characters.length,
                activation: 'softmax'
            })
        )

        // XXX: Finally, compile the model. The optimizer is used to define the backpropagation
        //      technique that should be used for training. We use the rmsProp to help tune the
        //      learning rate that we apply individually to each neuron to help learning.
        //      We use a categoricalCrossentropy loss model which is compatible with our softmax
        //      activation output.
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
    let sentenceIndices = Array.from(seed).map((e) =>
        this.characters.indexOf(e)
    )

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
            this.characters.length
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
        generated += this.characters[winnerIndex]
        sentenceIndices = sentenceIndices.slice(1)
        sentenceIndices.push(winnerIndex)
    }
    // console.log(`Generated text (temperature=${temperature}):\n ${generated}\n`)
    return generated
}

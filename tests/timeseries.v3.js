import * as tf from '@tensorflow/tfjs-node-gpu'

function createGRUModel(vocabSize, maxSequenceLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.gru({
            units: 32,
            inputShape: [maxSequenceLength, 1],
            returnSequences: true // Return the output sequences
        })
    )

    model.add(
        tf.layers.timeDistributed({
            layer: tf.layers.dense({ units: vocabSize, activation: 'softmax' }) // Predicting the probability distribution over the vocabulary
        })
    )

    return model
}

const vocab = Array.from(
    new Set(
        `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\\\`;:(){}[]<>#*^%$@~+-=_|/\n `
    )
)
const inputTexts = ['hello ', 'how are ', 'the weather ']

const outputTexts = ['world ', 'you doing? ', 'is nice ']

function preprocessData(texts, vocab, maxSequenceLength) {
    const inputIndices = texts.map((text) => {
        const chars = text.split('')
        const indices = chars.map((char) => vocab.indexOf(char))

        // Pad inputs to maxSequenceLength
        const padding = new Array(maxSequenceLength - indices.length).fill(0)
        const paddedIndices = indices.concat(padding)

        return paddedIndices.map((i) => [i])
    })

    return inputIndices
}

const maxSequenceLength = Math.max(...inputTexts.map((t) => t.length))

const inputIndices = preprocessData(inputTexts, vocab, maxSequenceLength)
const xTensor = tf.tensor3d(inputIndices)

const outputIndices = preprocessData(outputTexts, vocab, maxSequenceLength)
const flatOutputIndices = outputIndices.flat()
console.log(flatOutputIndices)
const yTensor = tf
    .oneHot(tf.tensor1d(flatOutputIndices.flat(), 'int32'), vocab.length)
    .reshape([inputTexts.length, maxSequenceLength, vocab.length])

// Create and compile the model
const model = createGRUModel(vocab.length, maxSequenceLength)
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' })

async function trainModel() {
    await model.fit(xTensor, yTensor, {
        epochs: 1000,
        verbose: 0
    })

    // Predict
    const prediction = model.predict([
        tf.tensor3d(preprocessData(['hello '], vocab, maxSequenceLength))
    ])

    // Convert predictions to indices (use argMax for categoricalCrossentropy)
    const predictedIndices = prediction.argMax(-1).dataSync()

    // Map indices to characters
    const predictedText = Array.from(predictedIndices)
        .map((index) => vocab[index])
        .join('')
    console.log('Predicted text:', predictedText)
}

trainModel()

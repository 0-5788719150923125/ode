import * as tf from '@tensorflow/tfjs-node-gpu'

function createGRUModel(vocabSize, inputLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.gru({
            units: 32,
            inputShape: [inputLength, 1],
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
const inputText = 'hello '
const outputText = 'world '

// Convert characters to indices
const inputIndices = inputText
    .split('')
    .map((char) => vocab.indexOf(char))
    .map((i) => [i])
const outputIndices = outputText.split('').map((char) => vocab.indexOf(char))

// Prepare input tensor
const xTensor = tf.tensor3d([inputIndices])

// Prepare target tensor as one-hot encoded vectors
const yTensor = tf
    .oneHot(tf.tensor1d(outputIndices, 'int32'), vocab.length)
    .reshape([1, outputIndices.length, vocab.length])

// Create and compile the model
const model = createGRUModel(vocab.length, inputText.length)
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' })

async function trainModel() {
    await model.fit(xTensor, yTensor, {
        epochs: 1000,
        verbose: 0
    })

    // Predict
    const prediction = model.predict(xTensor)

    // Convert predictions to indices (use argMax for categoricalCrossentropy)
    const predictedIndices = prediction.argMax(-1).dataSync()

    // Map indices to characters
    const predictedText = Array.from(predictedIndices)
        .map((index) => vocab[index])
        .join('')
    console.log('Predicted text:', predictedText)
}

trainModel()

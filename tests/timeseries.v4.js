import * as tf from '@tensorflow/tfjs-node-gpu'

function createGRUModel(vocabSize, maxSequenceLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: 16,
            maskZero: true
        })
    )

    model.add(
        tf.layers.gru({
            units: 32,
            returnSequences: true
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

const maxSequenceLength = Math.max(...inputTexts.map((t) => t.length))

function preprocessData(texts, vocab, maxSequenceLength) {
    return texts.map((text) => {
        const chars = text.split('')
        const indices = chars.map((char) => vocab.indexOf(char))
        const padding = new Array(maxSequenceLength - indices.length).fill(0)
        return indices.concat(padding)
    }) // This returns a 2D array: Array<Array<number>>, which is what we want
}

const inputIndices = preprocessData(inputTexts, vocab, maxSequenceLength)
console.log(inputIndices)
const xTensor = tf.tensor2d(inputIndices, [
    inputIndices.length,
    maxSequenceLength
]) // Use tensor2d instead of tensor3d

const outputIndices = preprocessData(outputTexts, vocab, maxSequenceLength)

const flatOutputIndices = outputIndices.flat().flat()

const yTensor = tf
    .oneHot(tf.tensor1d(flatOutputIndices, 'int32'), vocab.length)
    .reshape([inputTexts.length, maxSequenceLength, vocab.length])

console.log(xTensor)
console.log(yTensor)

// Create and compile the model
const model = createGRUModel(vocab.length, maxSequenceLength)
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' })

async function trainModel() {
    await model.fit(xTensor, yTensor, {
        epochs: 1000,
        verbose: 0
    })

    // Predict
    // Adjust the prediction input to match the expected 2D shape
    const predictionInput = preprocessData(['hello '], vocab, maxSequenceLength)
    const predictionTensor = tf.tensor2d(predictionInput, [
        predictionInput.length,
        maxSequenceLength
    ]) // Ensure this matches the training input shape

    // Predict using the adjusted 2D tensor
    const prediction = model.predict(predictionTensor)

    // Convert predictions to indices (use argMax for categoricalCrossentropy)
    const predictedIndices = prediction.argMax(-1).dataSync()

    // Map indices to characters
    const predictedText = Array.from(predictedIndices)
        .map((index) => vocab[index])
        .join('')
    console.log('Predicted text:', predictedText)
}

trainModel()
import * as tf from '@tensorflow/tfjs-node-gpu'

function createGRUModel(vocabSize) {
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

// Create and compile the model
const model = createGRUModel(vocab.length)
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' })

console.log(model.summary())
const inputTexts = ['hello ', 'how are ', 'the weather ']
const outputTexts = ['world ', 'you doing? ', 'is nice ']

const maxSequenceLength = 32

// function shuffleArray(array) {
//     for (let i = array.length - 1; i > 0; i--) {
//         const j = Math.floor(Math.random() * (i + 1))
//         ;[array[i], array[j]] = [array[j], array[i]] // Swap elements
//     }
//     return array
// }

function preprocessData(
    texts,
    numDuplicates = 0,
    vocab,
    maxSequenceLength,
    paddingSide = 'left'
) {
    let duplicates = texts
    if (numDuplicates > 0) {
        duplicates = Array.from({ length: 100 }, () => texts).flat()
    }

    // console.log(duplicates)
    return texts.map((text) => {
        const chars = text.split('')
        const indices = chars.map((char) => vocab.indexOf(char))
        const padding = new Array(maxSequenceLength - indices.length).fill(0)
        if (paddingSide === 'left') {
            return padding.concat(indices)
        } else if (paddingSide === 'right') {
            return indices.concat(padding)
        } else {
            while (indices.length < maxSequenceLength) {
                if (Math.random() < 0.5) {
                    indices.push(0)
                } else {
                    indices.unshift(0)
                }
            }
            return indices
        }
    }) // This returns a 2D array: Array<Array<number>>, which is what we want
}

const inputIndices = preprocessData(
    inputTexts,
    10000,
    vocab,
    maxSequenceLength,
    'left'
)
const outputIndices = preprocessData(
    outputTexts,
    10000,
    vocab,
    maxSequenceLength,
    'right'
)
console.log(inputIndices.slice(0, 3))
console.log(outputIndices.slice(0, 3))

const xTensor = tf.tensor2d(inputIndices, [
    inputIndices.length,
    maxSequenceLength
])

const ySequences = outputIndices.map((sequence) =>
    tf.oneHot(tf.tensor1d(sequence, 'int32'), vocab.length)
)

// Then, we stack these tensors together to create a single 3D tensor
// The shape of the tensor will be [numSequences, sequenceLength, vocabSize]
const yTensor = tf.stack(ySequences)

console.log(xTensor)
console.log(yTensor)

async function trainModel() {
    let step = 0
    await model.fit(xTensor, yTensor, {
        epochs: 10000,
        verbose: 0,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                step++
                if (step % 10 === 0) {
                    console.log(`STEP=${step}, LOSS=${logs.loss}`)
                }
                if (step % 100 === 0) {
                    for (const text of [
                        inputTexts[0],
                        inputTexts[1],
                        inputTexts[2]
                    ]) {
                        const predictionInput = preprocessData(
                            [text],
                            0,
                            vocab,
                            maxSequenceLength
                        )
                        const predictionTensor = tf.tensor2d(predictionInput, [
                            predictionInput.length,
                            maxSequenceLength
                        ])

                        const prediction = model.predict(predictionTensor)

                        const predictedIndices = prediction
                            .argMax(-1)
                            .dataSync()

                        const predictedText = Array.from(predictedIndices)
                            .map((index) => vocab[index])
                            .join('')
                        console.log('Predicted text:', predictedText)
                    }
                }
            }
        }
    })
}

trainModel()

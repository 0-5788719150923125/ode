import * as tf from '@tensorflow/tfjs-node-gpu'
import { shaks13 } from '../src/data.js'
// import pkg from '@tensorflow/tfjs-layers/dist/losses.js'
// const { sparseCategoricalCrossentropy } = pkg

const batchSize = 64
const maxSequenceLength = 100

function createGRUModel(vocabSize, batchSize, maxSequenceLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.embedding({
            inputDim: vocabSize,
            batchInputShape: [batchSize, maxSequenceLength],
            outputDim: 128,
            maskZero: true
        })
    )

    model.add(
        tf.layers.lstm({
            units: 128,
            returnSequences: true,
            stateful: false,
            returnState: false
        })
    )

    // model.add(
    //     tf.layers.lstm({
    //         units: 256,
    //         returnSequences: true,
    //         stateful: true,
    //         returnState: false
    //     })
    // )

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
const model = createGRUModel(vocab.length, batchSize, maxSequenceLength)
model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy' })

console.log(model.summary())

const { input, output } = sliceStringWithOverlap(shaks13, maxSequenceLength)

// function sliceString(inputString, length) {
//     const input = []
//     const output = []

//     for (let i = 0; i < inputString.length - length; i++) {
//         input.push(inputString.slice(i, i + length))
//         output.push(inputString.slice(i + 1, i + length + 1))
//     }

//     return { input, output }
// }

function sliceStringWithOverlap(inputString, length) {
    const input = []
    const output = []
    const step = Math.floor(length / 2)

    for (let i = 0; i < inputString.length - length; i += step) {
        const inputSlice = inputString.slice(i, i + length)
        const outputSlice = inputString.slice(i + step, i + step + length)

        input.push(inputSlice)
        output.push(outputSlice)
    }

    return { input, output }
}

function preprocessData(texts, vocab, maxSequenceLength, paddingSide = 'left') {
    return texts.map((text) => {
        const chars = text.split('')
        const indices = chars.map((char) => vocab.indexOf(char))
        const padding = new Array(maxSequenceLength - indices.length).fill(0)
        while (indices.length < maxSequenceLength) {
            if (paddingSide === 'left') {
                return padding.concat(indices)
            } else if (paddingSide === 'right') {
                return indices.concat(padding)
            } else {
                if (Math.random() < 0.5) {
                    indices.push(0)
                } else {
                    indices.unshift(0)
                }
            }
            return indices
        }
        return indices
    })
}
console.log(input.slice(0, 5))
console.log(output.slice(0, 5))
const inputIndices = preprocessData(input, vocab, maxSequenceLength, 'left')
const outputIndices = preprocessData(output, vocab, maxSequenceLength, 'left')
// console.log(inputIndices.slice(0, 3))
// console.log(outputIndices.slice(0, 3))

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
        verbose: 1,
        batchSize: batchSize,
        shuffle: false,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                step++
                if (step % 10 === 0) {
                    console.log(`STEP=${step}, LOSS=${logs.loss}`)
                }
                if (step % 100 === 0) {
                    model.resetStates()
                    for (const text of ['a', 'b', 'c']) {
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
                    model.resetStates()
                }
            }
        }
    })
}

trainModel()

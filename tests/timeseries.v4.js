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
model.compile({ optimizer: 'adam', loss: [tf.losses.softmaxCrossEntropy] })

const inputTexts = ['hello ', 'how are ', 'the weather ']
const outputTexts = ['world ', 'you doing? ', 'is nice ']

const maxSequenceLength = 32

function preprocessData(texts, vocab, maxSequenceLength, paddingSide = 'left') {
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
    vocab,
    maxSequenceLength,
    'left'
)
const outputIndices = preprocessData(
    outputTexts,
    vocab,
    maxSequenceLength,
    'left'
)
console.log(inputIndices)
console.log(outputIndices)

const xTensor = tf.tensor2d(inputIndices, [
    inputIndices.length,
    maxSequenceLength
])

const flatOutputIndices = outputIndices.flat().flat()

const yTensor = tf
    .oneHot(tf.tensor1d(flatOutputIndices, 'int32'), vocab.length)
    .reshape([inputTexts.length, maxSequenceLength, vocab.length])

console.log(xTensor)
console.log(yTensor)

async function trainModel() {
    let step = 0
    await model.fit(xTensor, yTensor, {
        epochs: 2500,
        verbose: 0,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                step++
                console.log(`STEP=${step}, LOSS=${logs.loss}`)
                if (step % 100 === 0) {
                    for (const text of inputTexts) {
                        const predictionInput = preprocessData(
                            [text],
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

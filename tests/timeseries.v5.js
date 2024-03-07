import * as tf from '@tensorflow/tfjs-node-gpu'
import { shaks13 } from '../src/data.js'
import { preprocessData } from '../src/utils.js'

const batchSize = 64
const maxSequenceLength = 100

function createGRUModel(vocabSize, batchSize, maxSequenceLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.embedding({
            inputDim: vocabSize,
            inputLength: maxSequenceLength,
            outputDim: 128,
            maskZero: true
        })
    )

    model.add(
        tf.layers.gru({
            units: 128,
            returnSequences: true,
            stateful: false,
            returnState: false
        })
    )

    // model.add(tf.layers.repeatVector(batchSize))

    model.add(
        tf.layers.gru({
            units: 128,
            returnSequences: true,
            stateful: false,
            returnState: false
        })
    )

    model.add(
        tf.layers.timeDistributed({
            layer: tf.layers.dense({ units: vocabSize, activation: 'softmax' })
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
const model = createGRUModel(vocab.length, batchSize, maxSequenceLength / 2)
model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy' })

console.log(model.summary())

const dataGenerator = stringSampler(maxSequenceLength, shaks13)
const dataset = tf.data.generator(
    createBatchGenerator(dataGenerator, vocab, batchSize, maxSequenceLength)
)

async function trainModel() {
    let step = 0
    await model.fitDataset(dataset, {
        epochs: 1,
        verbose: 0,
        batchSize: batchSize,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                step++
                if (step % 10 === 0) {
                    console.log(`STEP=${step}, LOSS=${logs.loss}`)
                }
                if (step % 100 === 0) {
                    const sample = dataGenerator.next().value
                    const textIndices = preprocessData(
                        sample,
                        vocab,
                        maxSequenceLength,
                        'left'
                    )
                    const xs = textIndices.slice(0, maxSequenceLength / 2)
                    // Adjust for a single example prediction
                    const xsTensor = tf.tensor2d(
                        [xs],
                        [1, maxSequenceLength / 2],
                        'int32'
                    ) // Note the use of [1, ...] for the shape

                    const prediction = model.predict(xsTensor)
                    const predictedIndices = prediction.argMax(-1).dataSync()

                    const predictedText = Array.from(predictedIndices)
                        .map((index) => vocab[index])
                        .join('')
                    console.log('Predicted text:', predictedText)
                }
            }
        }
    })
}

trainModel()

function* stringSampler(sampleLen, str = shaks13) {
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

function createBatchGenerator(
    dataGenerator,
    vocab,
    batchSize,
    inputLength,
    predictLength
) {
    return function* () {
        yield* batchGenerator(
            dataGenerator,
            vocab,
            batchSize,
            inputLength,
            predictLength
        )
    }
}

function* batchGenerator(dataGenerator, vocab, batchSize, inputLength) {
    while (true) {
        let xsArray = []
        let ysArray = []

        for (let i = 0; i < batchSize; ++i) {
            const text = dataGenerator.next().value
            // const sample = text.slice(0, randomBetween(1, inputLength))
            const sample = text

            const textIndices = preprocessData(
                sample,
                vocab,
                inputLength,
                'left'
            )

            // create input sequence
            const xs = textIndices.slice(0, inputLength / 2)

            // predict the last character index
            const ys = textIndices.slice(inputLength / 2, inputLength)

            xsArray.push(xs)
            ysArray.push(ys)
        }

        const xsTensor = tf.tensor2d(
            xsArray,
            [batchSize, inputLength / 2],
            'int32'
        )

        const ysFlat = ysArray.flat()
        const ysTensor = tf
            .oneHot(tf.tensor1d(ysFlat, 'int32'), vocab.length)
            .reshape([batchSize, inputLength / 2, vocab.length])

        yield { xs: xsTensor, ys: ysTensor }
    }
}

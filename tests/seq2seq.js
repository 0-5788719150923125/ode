import * as tf from '@tensorflow/tfjs-node-gpu'
import { shaks13 } from '../src/data.js'
import { emaGenerator, preprocessData, randomBetween } from '../src/utils.js'

const batchSize = 64
const maxSequenceLength = 100

const vocab = Array.from(
    new Set(
        `�¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\\\`;:(){}[]<>#*^%$@~+-=_|/\n `
    )
)

const dataGenerator = sequentialStringSampler(maxSequenceLength, shaks13)
const dataset = tf.data.generator(
    createBatchGenerator(dataGenerator, vocab, batchSize, maxSequenceLength)
)

function createGRUModel(vocabSize, batchSize, maxSequenceLength) {
    const model = tf.sequential()

    model.add(
        tf.layers.embedding({
            inputDim: vocabSize,
            inputLength: maxSequenceLength,
            batchInputShape: [batchSize, maxSequenceLength],
            outputDim: 256,
            maskZero: true
        })
    )

    model.add(
        tf.layers.bidirectional({
            layer: tf.layers.gru({
                units: 128,
                returnSequences: true,
                stateful: true,
                returnState: false
            }),
            mergeMode: 'concat'
        })
    )

    // model.add(
    //     tf.layers.repeatVector({
    //         n: maxSequenceLength
    //     })
    // )

    model.add(
        tf.layers.bidirectional({
            layer: tf.layers.gru({
                units: 128,
                returnSequences: true,
                stateful: true,
                returnState: false
            }),
            mergeMode: 'concat'
        })
    )

    model.add(
        tf.layers.timeDistributed({
            layer: tf.layers.dense({ units: vocabSize, activation: 'softmax' })
        })
    )

    return model
}

// Create and compile the model
const model = createGRUModel(vocab.length, batchSize, maxSequenceLength / 2)
model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy' })

console.log(model.summary())

async function trainModel() {
    let step = 0
    const timer = trainingTimer()
    const ema = emaGenerator()
    ema.next()
    makePrediction(0)
    await model.fitDataset(dataset, {
        epochs: 1,
        verbose: 0,
        batchSize: batchSize,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                step++
                model.resetStates()
                if (step % 10 === 0) {
                    const elapsedTime = timer.next().value
                    const updatedEma = ema.next(logs.loss).value
                    console.log(
                        `STEP=${step}, ELAPSED=${elapsedTime.toFixed(2)}Hrs, EMA=${updatedEma.toFixed(5)}, LOSS=${logs.loss}`
                    )
                }
                if (step % 100 === 0) {
                    console.log('GREEDY:')
                    makePrediction(0)
                    model.resetStates()
                    // console.log('TEMPERATURE (0.1):')
                    // makePrediction(0.1)
                    // model.resetStates()
                    // console.log('TEMPERATURE (0.3):')
                    // makePrediction(0.3)
                    // model.resetStates()
                    // console.log('TEMPERATURE (0.7):')
                    // makePrediction(0.7)
                }
            }
        }
    })
}

trainModel()

function* trainingTimer() {
    const startTime = new Date() // Mark the start time

    while (true) {
        const currentTime = new Date() // Get the current time on each call
        const elapsedTime = (currentTime - startTime) / (1000 * 60 * 60) // Calculate elapsed time in hours
        yield elapsedTime // Yield the elapsed time in hours
    }
}

function* sequentialStringSampler(sampleLen, str) {
    let index = 0 // Start from the first character
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0 // Loop back to the start if there's not enough room left for a full sample
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index++ // Move to the next character for the start of the next substring
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

            // const numTokensShifted = randomBetween(1, 6)
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

function makePrediction(temperature = 1.0) {
    const sample = dataGenerator.next().value
    const textIndices = preprocessData(sample, vocab, maxSequenceLength, 'left')
    const xs = textIndices.slice(0, maxSequenceLength / 2)
    const xsTensor = tf.tensor2d([xs], [1, maxSequenceLength / 2], 'int32')

    // Predict with the model
    const prediction = model.predict(xsTensor)

    // Squeeze to remove batch dimension since batch size is 1
    const squeezedPred = prediction.squeeze()

    let predictedSequence = []
    for (let i = 0; i < squeezedPred.shape[0]; i++) {
        const timestepPred = squeezedPred.slice([i, 0], [1, -1])

        let sampledIndex = greedySampling(timestepPred)
        // if (temperature === 0) {
        //     sampledIndex = greedySampling(timestepPred)
        // } else {
        //     sampledIndex = temperatureSampling(timestepPred, temperature)
        // }

        predictedSequence.push(vocab[sampledIndex])
    }

    const predictedText = predictedSequence.join('')
    console.log(`input: ${sample}`)
    console.log(`output: ${predictedText}`)
}

// Adjust greedySampling to work with single timestep predictions
function greedySampling(preds) {
    // Assuming preds is a single timestep prediction
    return preds.argMax(-1).dataSync()[0] // Take the first (and only) item
}

// Adjust temperatureSampling to directly return the sampled index
function temperatureSampling(preds, temperature = 1.0) {
    // Adjust the predictions by temperature
    let logits = preds.log()
    logits = logits.div(tf.scalar(temperature))
    let expPreds = logits.exp()
    let probs = expPreds.div(expPreds.sum())
    // let sumProbs = probs.sum().dataSync()[0]

    // // Print the sum of probabilities for debugging
    // console.log(`Sum of probabilities: ${sumProbs}`)

    // Sample from the adjusted probabilities
    return tf.multinomial(probs, 1).dataSync()[0]
}

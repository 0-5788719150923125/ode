import * as tfjs from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node-gpu'

let tf = tfjs

;(async function () {
    console.log('getting backend')
    await tf.ready()
    await tf.setBackend('tensorflow')
})()

import { shaks13 } from '../src/data.js'

const text = shaks13

const createModel = (lstmLayerSize, sampleLen, charSetSize, learningRate) => {
    const model = lstmLayerSize.reduce((mdl, lstmLayerSize, i, orig) => {
        mdl.add(
            tf.layers.lstm({
                units: lstmLayerSize,
                returnSequences: i < orig.length - 1,
                inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
            })
        )
        return mdl
    }, tf.sequential())
    model.add(
        tf.layers.dense({
            units: charSetSize,
            activation: 'softmax'
        })
    )
    model.compile({
        optimizer: tf.train.rmsprop(learningRate),
        loss: 'categoricalCrossentropy'
    })

    return model
}

const nextDataEpoch = (
    textLength,
    sampleLen,
    sampleStep,
    charSetSize,
    textIndices,
    numExamples
) => {
    const trainingIndices = []

    for (let i = 0; i < textLength - sampleLen - 1; i += sampleStep) {
        trainingIndices.push(i)
    }

    tf.util.shuffle(trainingIndices)

    const xsBuffer = new tf.TensorBuffer([numExamples, sampleLen, charSetSize])

    const ysBuffer = new tf.TensorBuffer([numExamples, charSetSize])

    for (let i = 0; i < numExamples; ++i) {
        const beginIndex = trainingIndices[i % trainingIndices.length]
        for (let j = 0; j < sampleLen; ++j) {
            xsBuffer.set(1, i, j, textIndices[beginIndex + j])
        }
        ysBuffer.set(1, i, textIndices[beginIndex + sampleLen])
    }

    return [xsBuffer.toTensor(), ysBuffer.toTensor()]
}

const generate = (
    model,
    seed,
    sampleLen,
    charSetSize,
    charSet,
    displayLength,
    temperature
) => {
    let sentenceIndices = Array.from(seed).map((e) => charSet.indexOf(e))

    let generated = ''

    let totalDuration = 0
    let count = 0

    while (generated.length < displayLength) {
        const startTime = performance.now()

        const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize])

        ;[...Array(sampleLen)].map((_, i) =>
            inputBuffer.set(1, 0, i, sentenceIndices[i])
        )

        const input = inputBuffer.toTensor()
        const output = model.predict(input)

        const [winnerIndex] = tf.tidy(() =>
            tf
                .multinomial(
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

        input.dispose()
        output.dispose()

        generated += charSet[winnerIndex]
        sentenceIndices = sentenceIndices.slice(1)
        sentenceIndices.push(winnerIndex)

        const endTime = performance.now()
        const duration = endTime - startTime
        totalDuration += duration
        count++
    }

    console.log(`Generated text (temperature=${temperature}):\n${generated}\n`)
    console.log(`Average duration per iteration: ${totalDuration / count} ms`)
}

;(async () => {
    const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
    const sampleStep = 3 // number of characters to jump between segments of input text
    const learningRate = 1e-2 // higher values lead to faster convergence, but more errors
    const epochs = 150 // the total number of times to update the training weights
    const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data
    const displayLength = 120 // how many characters you want to generate after training
    const lstmLayerSize = [16, 16, 16] // the configuration of eah sequential lstm network
    const temperatures = [0, 0.3, 0.7]

    const charSet = Array.from(new Set(Array.from(text)))
    const { length: charSetSize } = charSet

    const model = createModel(
        lstmLayerSize,
        sampleLen,
        charSetSize,
        learningRate
    )

    model.summary()

    const textIndices = new Uint16Array(
        Array.from(text).map((e) => charSet.indexOf(e))
    )

    const startIndex = Math.round(Math.random() * (text.length - sampleLen - 1))

    const seed = text.slice(startIndex, startIndex + sampleLen)

    for (let i = 0; i < epochs; ++i) {
        const [xs, ys] = nextDataEpoch(
            text.length,
            sampleLen,
            sampleStep,
            charSetSize,
            textIndices,
            examplesPerEpoch
        )

        await model.fit(xs, ys, {
            epochs: 1,
            batchSize,
            validationSplit,
            callbacks: {
                onTrainBegin: () => {
                    console.log(`Epoch ${i + 1} of ${epochs}:`)
                },
                onTrainEnd: () =>
                    temperatures.map((temp) =>
                        generate(
                            model,
                            seed,
                            sampleLen,
                            charSetSize,
                            charSet,
                            displayLength,
                            temp
                        )
                    )
            }
        })

        xs.dispose()
        ys.dispose()
    }
})()

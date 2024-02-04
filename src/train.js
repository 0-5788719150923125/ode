import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'

function randomString(
    len,
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
) {
    let text = ''
    for (let i = 0; i < len; i++) {
        text += chars.charAt(Math.floor(Math.random() * chars.length))
    }
    return text
}

function* infiniteNumbers() {
    let i = 0
    while (i < 500) {
        i++
        yield randomString(10000)
    }
}

export async function trainModel(model, dataGenerator = infiniteNumbers) {
    // XXX: .
    const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
    const sampleStep = 3 // number of characters to jump between segments of input text
    const epochs = 150 // the total number of times to update the training weights
    const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data
    const displayLength = 120 // how many characters you want to generate after training
    const temperatures = [0, 0.25, 0.5, 0.75, 1]

    const data = dataGenerator()
    // console.log(data.next().value)

    // XXX: Fetch the text data to sample from.
    //   const { data: text } = await axios({
    //     method: 'get',
    //     url
    //   })

    let text = data.next().value

    // XXX: Fetch all unique characters in the dataset. (quickly!)
    const charSet = Array.from(new Set(Array.from(text)))
    const { length: charSetSize } = charSet

    // XXX: Convert the total input character text into the corresponding indices in the
    //      charSet. This is how we map consistently between character data and numeric
    //      neural network dataj

    // XXX: Pick a random position to start in the dataset. (Note that we choose an index
    //      which cannot exceed the minimum size of our sampleLength - 1).
    // const startIndex = Math.round(Math.random() * (text.length - sampleLen - 1))

    // XXX: Create the seed data which we'll use to initialize the network.
    // const seed = text.slice(startIndex, startIndex + sampleLen)

    const textIndices = new Uint16Array(
        Array.from(text).map((e) => charSet.indexOf(e))
    )

    for (let i = 0; i < epochs; ++i) {
        const [xs, ys] = dataToTensor(
            text.length,
            sampleLen,
            sampleStep,
            charSetSize,
            textIndices,
            examplesPerEpoch
        )

        // XXX: Fit the model and hold up iteration of the for loop
        //      until it is finished.
        await model.fit(xs, ys, {
            epochs: 1,
            batchSize,
            validationSplit,
            callbacks: {
                onTrainBegin: () => {
                    console.log(`Epoch ${i + 1} of ${epochs}:`)
                }
                // onTrainEnd: () =>
                //   temperatures.map((temp) =>
                //     generate(
                //       model,
                //       seed,
                //       sampleLen,
                //       charSetSize,
                //       charSet,
                //       displayLength,
                //       temp
                //     )
                //   )
            }
        })

        xs.dispose()
        ys.dispose()
    }
}

const dataToTensor = (
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

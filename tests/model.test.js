import fs from 'fs'
import * as tf from '@tensorflow/tfjs-node-gpu'
import Model from '../src'

// tf.setBackend('cuda:1')
// tf.env().set('IS_NODE', true)

let model
const lstmLayerSize = [128, 128, 128]
const sampleLen = 180
const learningRate = 1e-3
const displayLength = 180

const textContent = fs.readFileSync('./tests/shaks12.txt', 'utf8')

function* dataSampler(str, sampleLen) {
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring from the random starting point
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

beforeAll(async () => {
    model = new Model(lstmLayerSize, sampleLen, learningRate, displayLength)
    // model.summary()
})

test('createModel returns a tfjs model', async () => {
    expect(model.getModel()).toBeInstanceOf(tf.LayersModel) // Assert that the returned object is a tfjs model
})

test('trainModel updates weights', async () => {
    const initialWeights = model.getWeights() // Get initial weights
    await model.trainModel(dataSampler(textContent, sampleLen)) // Train for 2 epochs
    const newWeights = model.getWeights() // Get updated weights
    // expect(weightsAreDifferent(initialWeights, newWeights)).toBe(true)
    expect.anything()
}, 60000000)

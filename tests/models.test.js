import * as tf from '@tensorflow/tfjs'
import Model from '../dist'

// tf.setBackend('cpu')
// tf.env().set('IS_NODE', true)

let model
const lstmLayerSize = [128, 128]
const sampleLen = 60
const learningRate = 1e-2

beforeAll(async () => {
    model = new Model(lstmLayerSize, sampleLen, learningRate)
    // model.summary()
})

test('createModel returns a tfjs model', async () => {
    expect(model.getModel()).toBeInstanceOf(tf.LayersModel) // Assert that the returned object is a tfjs model
})

test('trainModel updates weights', async () => {
    const initialWeights = model.getWeights() // Get initial weights
    await model.trainModel() // Train for 2 epochs
    const newWeights = model.getWeights() // Get updated weights
    expect(weightsAreDifferent(initialWeights, newWeights)).toBe(true)
    done()
}, 60000)

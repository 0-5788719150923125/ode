import * as tf from '@tensorflow/tfjs'
import ODE from '../src/index.js'

let net

beforeAll(async () => {
    net = await ODE({
        mode: 'train',
        backend: 'tensorflow',
        version: 3,
        batchSize: 1,
        gradientAccumulationSteps: 128,
        generateEvery: 256,
        sampleLength: 256,
        predictLength: 128,
        saveEvery: 0,
        corpus: null,
        contextLength: 512,
        clipValue: 1.0
    })
})

test('net can initialize', async () => {
    await net.init()
})

test('net contains a tfjs model', async () => {
    expect(net.model).toBeInstanceOf(tf.LayersModel)
})

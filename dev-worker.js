import ODE from './src/index.js'
import { stringSampler } from './src/utils.js'

// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs'
// Add the WebGPU backend to the global backend registry.
// import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-cpu'

onmessage = async function (event) {
    // Set the backend to WebGPU and wait for the module to be ready.
    // await tf.ready()
    await tf.setBackend('cpu')

    console.log(event.data)

    const net = new ODE({
        layout: [128, 128, 128],
        learningRate: 1e-3,
        predictLength: 100,
        inputLength: 300,
        embeddingDimensions: 256
    })

    console.log(net.model.summary())

    const sampleLen = 30
    const dataset = stringSampler(sampleLen)
    const batchSize = 2

    await net.train(dataset, batchSize)
}

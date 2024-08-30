// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs-node'
import fs from 'fs'

// Create dummy data
const numSamples = 1000
const inputDim = 5
const outputDim = 1

const generateDummyData = () => {
    const x = tf.randomNormal([numSamples, inputDim])
    const w = tf.randomNormal([inputDim, 1])
    const b = tf.scalar(0.5)
    const noise = tf.randomNormal([numSamples, 1], 0, 0.1)
    const y = x.matMul(w).add(b).add(noise)
    return [x, y]
}

const [xData, yData] = generateDummyData()

// Define the model using Functional API
const createModel = () => {
    const input = tf.input({ shape: [inputDim] })
    const dense1 = tf.layers
        .dense({ units: 10, activation: 'relu' })
        .apply(input)
    const output = tf.layers.dense({ units: outputDim }).apply(dense1)
    return tf.model({ inputs: input, outputs: output })
}

let model = createModel()

// Define loss function and optimizer
const loss = tf.losses.meanSquaredError
const learningRate = 0.01
const optimizer = tf.train.adam(learningRate)

// Custom training loop
const trainModel = async (epochs, initialEpoch = 0) => {
    for (let epoch = initialEpoch; epoch < initialEpoch + epochs; epoch++) {
        tf.tidy(() => {
            const predictions = model.predict(xData)
            const currentLoss = loss(yData, predictions)
            const grads = tf.variableGrads(() =>
                loss(yData, model.predict(xData))
            )
            optimizer.applyGradients(grads.grads)
            console.log(
                `Epoch ${epoch + 1}, Loss: ${currentLoss.dataSync()[0]}`
            )
        })
        await tf.nextFrame()
    }
}

// Function to save the model
const saveModel = async () => {
    await model.save('file://my-model', { includeOptimizer: true })
    console.log('Model saved')
}

// Function to load the model
const loadModel = async () => {
    model = await tf.loadLayersModel('file://my-model/model.json')
    console.log('Model loaded')
}

// Train, save, load, and continue training
const runExperiment = async () => {
    console.log('Initial training:')
    await trainModel(25)

    await saveModel()

    // Create a new model instance to simulate a fresh start
    model = createModel()

    await loadModel()

    console.log('Continuing training after load:')
    await trainModel(25, 25)

    // Test the model
    const testInput = tf.randomNormal([1, inputDim])
    const prediction = model.predict(testInput)
    console.log('Final test prediction:', prediction.dataSync())
}

runExperiment().then(() => console.log('Experiment complete'))

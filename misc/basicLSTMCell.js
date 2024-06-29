import * as tf from '@tensorflow/tfjs'

const forgetBias = tf.scalar(1.0)
const lstmUnits = 128
const inputSize = 10
const batchSize = 32

// Initialize LSTM kernel and bias
const lstmKernel = tf.randomNormal([lstmUnits + inputSize, 4 * lstmUnits])
const lstmBias = tf.randomNormal([4 * lstmUnits])

// Initial cell state and hidden state
let c = tf.zeros([batchSize, lstmUnits])
let h = tf.zeros([batchSize, lstmUnits])

// Input data (random example)
const data = tf.randomNormal([batchSize, inputSize])

// Compute the next state and output
;[c, h] = tf.basicLSTMCell(forgetBias, lstmKernel, lstmBias, data, c, h)

console.log('New cell state shape:', c.shape)
console.log(c)
console.log('New hidden state shape:', h.shape)

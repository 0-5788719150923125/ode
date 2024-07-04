import * as tf from '@tensorflow/tfjs-node'

let X = tf.randomNormal([4096, 512])
let Y = tf.randomNormal([256, 512])

const iterations = 100
for (let i = 0; i < iterations; i++) {
    Y = tf.tidy(() => {
        const clonedW = Y.clone()
        Y.dispose()
        const YZ = tf.matMul(clonedW, X.transpose())
        return clonedW
    })
    console.log(tf.memory().numTensors)
}

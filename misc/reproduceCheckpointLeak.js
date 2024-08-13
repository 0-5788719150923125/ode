import fs from 'fs'
import * as tf from '@tensorflow/tfjs-node'

const model = tf.sequential()
model.add(tf.layers.dense({ units: 10, inputShape: [1] }))
model.add(tf.layers.dense({ units: 1 }))

model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
})

const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
const ys = tf.tensor2d([2, 4, 6, 8], [4, 1])

fs.mkdirSync('./saved_model', { recursive: true })

model.fit(xs, ys, {
    epochs: Infinity,
    verbose: 0,
    callbacks: {
        onEpochEnd: async (epoch, logs) => {
            console.clear()
            console.log(epoch)
            console.log(tf.memory())
            if (epoch % 1000 === 0 && epoch !== 0) {
                // tf.engine().startScope()
                await model.save(`file://saved_model`, {
                    includeOptimizer: true
                })
                // tf.engine().endScope()
            }
        }
    }
})

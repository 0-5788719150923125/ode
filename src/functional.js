// import * as tf from '@tensorflow/tfjs'

// function createExpert(inputShape, units) {
//     const input = tf.input({ shape: inputShape })
//     const output = tf.layers
//         .dense({
//             units: units,
//             kernelInitializer: 'glorotUniform',
//             useBias: true
//         })
//         .apply(input)
//     return tf.model({ inputs: input, outputs: output })
// }

// function createMoEModel(inputShape, expertConfig) {
//     const inputs = tf.input({ shape: inputShape })

//     // Create a set of expert models based on the configuration
//     const experts = expertConfig.map((config) =>
//         createExpert(inputShape.slice(1), config.units)
//     )

//     // Placeholder for expert selection logic; this could be replaced with
//     // a more dynamic mechanism based on inputs or trainable parameters
//     const selectedExpertIndex = 0 // Simplified selection logic for demonstration
//     const selectedExpertOutput = experts[selectedExpertIndex].apply(inputs)

//     // Further model definition could go here

//     return tf.model({ inputs: inputs, outputs: selectedExpertOutput })
// }

// // Example usage
// const expertConfig = [{ units: 64 }, { units: 32 }]
// const model = createMoEModel([null, 10, 16], expertConfig) // Assuming input shape of [batchSize, 10, 16]

import tf from '@tensorflow/tfjs-node' // or '@tensorflow/tfjs' if in browser

// Define the input layer
const inputs = tf.input({ shape: [10] })

// Add a dense layer
const denseLayer = tf.layers.dense({ units: 32, activation: 'relu' })
const denseOutput = denseLayer.apply(inputs)

// Add an output layer
const outputs = tf.layers
    .dense({ units: 1, activation: 'sigmoid' })
    .apply(denseOutput)

// Create the model
const model = tf.model({ inputs: inputs, outputs: outputs })

model.compile({
    optimizer: 'sgd',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
})

const xs = tf.randomNormal([100, 10]) // 100 samples, 10 features
const ys = tf.randomUniform([100, 1]).step() // 100 labels, binary (0 or 1)

console.log(model.summary())

async function trainModel() {
    const history = await model.fit(xs, ys, {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.2,
        verbose: 0,
        callbacks: {
            onBatchEnd: (batch, logs) => {
                console.log(logs)
            }
        }
    })

    console.log(history.history)
}

trainModel().catch(console.error)

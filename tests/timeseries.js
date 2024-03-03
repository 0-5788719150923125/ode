import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
        await tf.ready()
        await tf.setBackend('tensorflow')
    }
})()

// Define a model with a GRU layer
function createGRUModel() {
    const model = tf.sequential()

    // Add a GRU layer
    // Adjusting inputShape to [3, 1] to match the reshaped input
    model.add(
        tf.layers.gru({
            units: 5, // Adjust the number of units as necessary
            inputShape: [3, 1] // 3 timesteps with 1 feature each
        })
    )

    // Add a dense layer to produce the desired output shape
    model.add(tf.layers.dense({ units: 2 })) // Predicting a sequence of 2 values

    return model
}

// Create the model
const model = createGRUModel()

// Compile the model with an optimizer and loss function
model.compile({ optimizer: 'adam', loss: 'meanSquaredError' })

// Prepare the input data with an extra dimension for features
// Reshape or construct the input tensor to have shape [batch_size, timesteps, features]
const xTensor = tf.tensor3d([[[1], [2], [3]]]) // Shape: [1, 3, 1], representing 1 sequence of 3 timesteps with 1 feature each
const yTensor = tf.tensor2d([4, 5], [1, 2]) // Target output

// Train the model
async function trainModel() {
    await model.fit(xTensor, yTensor, {
        epochs: 100 // Adjust epochs as necessary
    })

    // Use the model to make predictions
    const output = model.predict(xTensor)
    output.print() // This should output predictions close to your target values after sufficient training
}

trainModel()

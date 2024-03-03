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
function createGRUModel(input, output) {
    const model = tf.sequential()

    // Add a GRU layer
    // Adjusting inputShape to [3, 1] to match the reshaped input
    model.add(
        tf.layers.gru({
            units: 32, // Adjust the number of units as necessary
            inputShape: [input.length, 1] // 3 timesteps with 1 feature each
        })
    )

    // Add a dense layer to produce the desired output shape
    model.add(tf.layers.dense({ units: output.length })) // Predicting a sequence of 2 values

    return model
}

// Prepare the input data with an extra dimension for features
// Reshape or construct the input tensor to have shape [batch_size, timesteps, features]
let input = [[1], [2], [3]]
let output = [4, 5]

// const vocab = Array.from(
//     new Set(
//         `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
//     )
// )
// var input = 'hello '
//     .split('')
//     .map((char) => vocab.indexOf(char))
//     .filter((index) => index !== -1)
//     .map((i) => [i])
// let output = 'wo'
//     .split('')
//     .map((char) => vocab.indexOf(char))
//     .filter((index) => index !== -1)

// console.log(input)
// console.log(output)

const xTensor = tf.tensor3d([input]) // Shape: [1, 3, 1], representing 1 sequence of 3 timesteps with 1 feature each
const yTensor = tf.tensor2d(output, [1, output.length]) // Target output

// Create the model
const model = createGRUModel(input, output)

// Compile the model with an optimizer and loss function
model.compile({ optimizer: 'adam', loss: 'meanSquaredError' })

// Train the model
async function trainModel() {
    await model.fit(xTensor, yTensor, {
        epochs: 100 // Adjust epochs as necessary
    })

    // Use the model to make predictions
    const output = model.predict(xTensor).dataSync()
    console.log(output)

    // Step 1: Round the predictions to the nearest integer to get indices
    const indices = output.map((value) => Math.round(value))

    // Step 2: Map indices to characters using the vocabulary
    const characters = indices.map((index) => {
        // Ensure the index is within the bounds of the vocabulary
        if (index >= 0 && index < vocab.length) {
            return vocab[index]
        }
        // Return a placeholder (e.g., "?") for indices outside the bounds
        return '?'
    })

    // Step 3: Concatenate characters to form the resulting string
    const outputText = characters.join('')

    console.log('Predicted text:', outputText)

    // output.print() // This should output predictions close to your target values after sufficient training
}

trainModel()

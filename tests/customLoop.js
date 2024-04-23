import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-webgl'
;(async function () {
    // await tf.setBackend('webgl')
    // Create a simple model
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

    // Compile the model with a simple loss function
    model.compile({ loss: [tf.losses.meanSquaredError], optimizer: 'sgd' })

    // Generate dummy data
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

    const preds = model.call(xs)
    const losses = model.loss[0](ys, preds[0])

    console.log(losses)
})()

// // Define the number of epochs and batch size
// const epochs = 100
// const batchSize = 2

// // Custom training loop
// async function train() {
//     for (let epoch = 0; epoch < epochs; epoch++) {
//         // Shuffle the data for each epoch
//         const indices = tf.util.createShuffledIndices(xs.shape[0])
//         const shuffledXs = xs.gather(indices)
//         const shuffledYs = ys.gather(indices)

//         // Iterate over the batches
//         for (let i = 0; i < shuffledXs.shape[0]; i += batchSize) {
//             const batchXs = shuffledXs.slice(i, i + batchSize)
//             const batchYs = shuffledYs.slice(i, i + batchSize)

//             // Train the model on the batch
//             await model.trainOnBatch(batchXs, batchYs)
//         }

//         // Print the loss for every 10 epochs
//         if ((epoch + 1) % 10 === 0) {
//             const loss = model.evaluate(xs, ys)
//             console.log(`Epoch ${epoch + 1}: Loss = ${loss.dataSync()}`)
//         }
//     }
// }

// // Start the training
// train().then(() => {
//     // Make predictions using the trained model
//     const preds = model.predict(xs)
//     preds.print()
// })

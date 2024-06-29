const model = tf.sequential()
model.add(
    tf.layers.dense({ units: 32, inputShape: [stateSize], activation: 'relu' })
)
model.add(tf.layers.dense({ units: actionSize, activation: 'softmax' }))

const optimizer = tf.train.adam(learningRate)

function trainAgent(state, action, reward, nextState) {
    // Perform a forward pass to get action probabilities
    const actionProbs = model.predict(state)

    // Calculate the loss based on the action taken and the reward received
    const loss = tf.scalar(reward).sub(actionProbs.gather(action))

    // Perform a backward pass to update the model's weights
    optimizer.minimize(() => loss)
}

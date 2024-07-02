// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs-node'

class SequenceModel {
    constructor(vocabSize, embeddingDim, sequenceLength) {
        this.model = tf.sequential()
        this.model.add(
            tf.layers.embedding({
                inputDim: vocabSize,
                outputDim: embeddingDim,
                inputLength: sequenceLength
            })
        )
        this.model.add(
            tf.layers.timeDistributed({
                layer: tf.layers.dense({
                    units: vocabSize,
                    activation: 'softmax'
                })
            })
        )
    }

    call(inputs) {
        return this.model.predict(inputs)
    }
}

async function train(model, inputs, targets, epochs = 10) {
    const optimizer = tf.train.adam(0.01)

    for (let epoch = 0; epoch < epochs; epoch++) {
        const { value, grads } = tf.variableGrads(() => {
            const logits = model.call(inputs)
            return tf.losses.softmaxCrossEntropy(targets, logits)
        })

        optimizer.applyGradients(grads)

        console.log(`Epoch ${epoch + 1}, Loss: ${await value.data()}`)
        value.dispose()
        await tf.nextFrame() // Allows the UI to update
    }
}

async function runExample() {
    const vocabSize = 1000
    const embeddingDim = 32
    const sequenceLength = 10
    const batchSize = 32
    const epochs = 1000

    const model = new SequenceModel(vocabSize, embeddingDim, sequenceLength)

    // Generate some random data
    const inputs = tf.randomUniform(
        [batchSize, sequenceLength],
        0,
        vocabSize - 1,
        'int32'
    )
    const targets = tf.randomUniform(
        [batchSize, sequenceLength, vocabSize],
        0,
        1
    )

    await train(model, inputs, targets, epochs)
}

runExample()

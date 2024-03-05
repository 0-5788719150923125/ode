import * as tf from '@tensorflow/tfjs'

class MixtureOfExpertsLayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.expertCount = config.expertCount || 2 // Number of experts
        this.units = config.units // Number of units in the gating and expert layers
    }

    build(inputShape) {
        // Gating mechanism to decide which expert to use for each sample
        this.gate = this.addWeight(
            'gate',
            [inputShape[inputShape.length - 1], this.expertCount],
            'float32',
            tf.initializers.glorotUniform({})
        )

        // Experts are simple Dense layers in this example
        this.experts = []
        for (let i = 0; i < this.expertCount; i++) {
            this.experts.push(
                this.addWeight(
                    `expert_${i}`,
                    [inputShape[inputShape.length - 1], this.units],
                    'float32',
                    tf.initializers.glorotUniform({})
                )
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const gateOutput = tf.softmax(tf.dot(inputs, this.gate.read()), 1) // Softmax to ensure output sums to 1
            let output = null

            for (let i = 0; i < this.expertCount; i++) {
                // Compute the output for each expert
                const expertOutput = tf.dot(inputs, this.experts[i].read())
                // Weight the output by the gating mechanism
                const weightedOutput = tf.mul(
                    expertOutput,
                    gateOutput.slice([0, i], [-1, 1])
                )

                if (output === null) {
                    output = weightedOutput
                } else {
                    output = tf.add(output, weightedOutput)
                }
            }

            return output
        })
    }

    // TensorFlow.js requires the getConfig method to save and load models
    getConfig() {
        const config = super.getConfig()
        config.expertCount = this.expertCount
        config.units = this.units
        return config
    }

    // Static method to help TensorFlow.js identify this class
    static get className() {
        return 'MixtureOfExpertsLayer'
    }
}

tf.serialization.registerClass(MixtureOfExpertsLayer)

class SimplifiedMoELayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.units = config.units
        this.expertCount = config.expertCount || 2
    }

    build(inputShape) {
        // Initialize gating mechanism
        this.gate = this.addWeight(
            'gate',
            [inputShape[inputShape.length - 1], this.expertCount],
            'float32',
            tf.initializers.glorotUniform({})
        )

        // Initialize experts
        this.experts = []
        for (let i = 0; i < this.expertCount; i++) {
            let expert = tf.layers.dense({
                units: this.units,
                activation: 'relu', // Example activation
                kernelInitializer: 'glorotUniform',
                useBias: true
            })
            expert.build(inputShape) // Manually set input shape
            this.experts.push(expert)
        }
    }

    call(inputs) {
        let gateScores = tf.matMul(inputs, this.gate.read())
        gateScores = tf.softmax(gateScores) // Ensure scores sum to 1

        // Example of simplifying by just using a weighted sum of experts
        let output = null
        for (let i = 0; i < this.expertCount; i++) {
            let expertOutput = this.experts[i].apply(inputs)
            let weightedOutput = tf.mul(
                expertOutput,
                gateScores.slice([0, i], [-1, 1])
            )
            if (output === null) {
                output = weightedOutput
            } else {
                output = tf.add(output, weightedOutput)
            }
        }
        return output
    }

    getConfig() {
        return {
            units: this.units,
            expertCount: this.expertCount
        }
    }

    static get className() {
        return 'SimplifiedMoELayer'
    }
}

tf.serialization.registerClass(SimplifiedMoELayer)

class SparseMixtureOfExpertsLayer extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.expertCount = config.expertCount || 2 // Number of experts
        this.units = config.units // Number of units in the gating and expert layers
        this.experts = []
    }

    build(inputShape) {
        this.gate = this.addWeight(
            'gate',
            [256, this.expertCount], // Adjust to match GRU's output size
            'float32',
            tf.initializers.glorotUniform({})
        )

        // Correctly size the expert layers
        const expertUnits = Math.floor(
            inputShape[inputShape.length - 1] / this.expertCount
        )

        for (let i = 0; i < this.expertCount; i++) {
            const expertLayer = tf.layers.dense({
                units: expertUnits, // Adjusted units
                kernelInitializer: 'glorotUniform',
                useBias: false
            })
            expertLayer.build(inputShape)
            this.experts.push(expertLayer)
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // Calculate gate scores for each expert
            const gateScores = []
            for (let i = 0; i < this.expertCount; i++) {
                const expertWeights = this.experts[i].getWeights()[0]
                gateScores.push(tf.matMul(inputs, expertWeights, false, true)) // transposeB=true
            }
            const gateScoresTensor = tf.stack(gateScores, (axis = 1))

            // Select the expert with the highest score
            const gateMaxIndices = tf.argMax(gateScoresTensor, 1)

            // Apply experts directly
            const expertOutputs = []
            for (let i = 0; i < this.expertCount; i++) {
                expertOutputs.push(this.experts[i].apply(inputs))
            }

            // Gather based on indices
            const selectedOutputs = tf.gatherND(
                tf.stack(expertOutputs, (axis = 1)),
                gateMaxIndices.expandDims(1),
                1
            )

            return selectedOutputs.squeeze([1])
        })
    }

    // TensorFlow.js requires the getConfig method to save and load models
    getConfig() {
        const config = super.getConfig()
        config.expertCount = this.expertCount
        config.units = this.units
        return config
    }

    static get className() {
        return 'SparseMixtureOfExpertsLayer'
    }
}

tf.serialization.registerClass(SparseMixtureOfExpertsLayer)

// Generate some mock data with corrected type casting for 'oneHot'
const xTrain = tf.randomNormal([1000, 10, 64]) // 1000 samples, 10 time steps, 64 features per step
const yIndices = tf.floor(tf.randomUniform([1000], 0, 10)).toInt() // Correctly cast to int32
const yTrain = tf.oneHot(yIndices, 10) // 1000 labels, 10 classes

// Define the model
const model = tf.sequential()
model.add(
    tf.layers.gru({ units: 256, returnSequences: false, inputShape: [10, 64] })
) // Note returnSequences set to false
model.add(new SparseMixtureOfExpertsLayer({ units: 128, expertCount: 2 }))
model.add(tf.layers.dense({ units: 10, activation: 'softmax' })) // Assuming a classification task with 10 classes

model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
})

console.log(model.summary())

// Train the model
model
    .fit(xTrain, yTrain, {
        epochs: 10,
        batchSize: 32,
        callbacks: tf.callbacks.earlyStopping({ patience: 3 })
    })
    .then((info) => {
        console.log('Training complete')
        console.log('Final accuracy:', info.history.acc)
    })
    .catch((error) => {
        console.error('Training failed', error)
    })

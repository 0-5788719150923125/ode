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

class SparseMixtureOfExpertsLayer extends tf.layers.Layer {
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
            const gateScores = tf.dot(inputs, this.gate.read()) // Compute gating scores for each expert
            const gateMaxIndices = tf.argMax(gateScores, 1) // Find the index of the max scoring expert for each input

            let output = tf.zerosLike(tf.dot(inputs, this.experts[0].read()))

            for (let i = 0; i < this.expertCount; i++) {
                // Create a mask that is 1 for inputs where this expert is the max, and 0 otherwise
                const expertMask = tf
                    .equal(gateMaxIndices, tf.scalar(i))
                    .asType('float32')
                // Expand dimensions to match the multiplication needs
                const expertMaskExpanded = expertMask.expandDims(-1)

                // Compute the output for this expert
                const expertOutput = tf.dot(inputs, this.experts[i].read())

                // Apply the mask to zero out inputs not meant for this expert
                const maskedExpertOutput = tf.mul(
                    expertOutput,
                    expertMaskExpanded
                )

                // Add this expert's contribution to the output
                output = tf.add(output, maskedExpertOutput)
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

    static get className() {
        return 'SparseMixtureOfExpertsLayer'
    }
}

// Register the custom layer
tf.serialization.registerClass(MixtureOfExpertsLayer)
tf.serialization.registerClass(SparseMixtureOfExpertsLayer)

// Example usage
const model = tf.sequential()
model.add(
    tf.layers.gru({ units: 256, returnSequences: true, inputShape: [10, 64] })
) // Example preceding layer
model.add(new SparseMixtureOfExpertsLayer({ units: 128, expertCount: 2 })) // Sparse MoE layer
model.add(tf.layers.dense({ units: 10 })) // Following layer for output

console.log(model.summary())

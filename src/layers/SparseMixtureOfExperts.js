import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SparseMixtureOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 128
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[0][inputShape[0].length - 1]

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.switchingDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingBias = this.addWeight(
            'gatingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const expertInputs = inputs.slice(1)
            inputs = inputs[0]

            // Gating network
            const gatingHidden = this.ops.applyDense(
                inputs,
                this.gatingHidden.read(),
                this.gatingHiddenBias.read()
            )
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(gatingHidden)
            const expertWeights = this.ops
                .applyDense(
                    activatedGate,
                    this.gatingKernel.read(),
                    this.gatingBias.read()
                )
                .softmax()

            // Randomly select a subset of experts
            const selectedExpertIndices = this.selectRandomExperts(expertInputs)

            // Slice the expert weights based on the selected expert indices
            const selectedExpertWeights = this.sliceExpertWeights(
                expertWeights,
                selectedExpertIndices
            )

            // Slice and combine selected expert outputs
            const selectedExpertOutputs = []
            selectedExpertIndices.map((expertIndex) => {
                selectedExpertOutputs.push(expertInputs[expertIndex])
            })

            // Combine expert outputs using weighted sum
            const combinedOutput = selectedExpertOutputs.reduce(
                (prev, curr, i) => {
                    const expertWeight = selectedExpertWeights.slice(
                        [0, 0, i],
                        [inputs.shape[0], inputs.shape[1], 1]
                    )
                    return prev.add(curr.mul(expertWeight))
                },
                tf.zeros(expertInputs[0].shape)
            )

            return combinedOutput
        })
    }

    selectRandomExperts(expertInputs) {
        const numExperts = expertInputs.length
        const expertIndices = tf.util.createShuffledIndices(numExperts)
        return expertIndices.slice(0, this.topK)
    }

    sliceExpertWeights(expertWeights, selectedExpertIndices) {
        const selectedWeights = []
        selectedExpertIndices.forEach((expertIndex) => {
            const expertSlice = expertWeights.slice(
                [0, 0, expertIndex],
                [expertWeights.shape[0], expertWeights.shape[1], 1]
            )
            selectedWeights.push(expertSlice)
        })
        return tf.concat(selectedWeights, -1)
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK
        }
    }
}

tf.serialization.registerClass(SparseMixtureOfExperts)

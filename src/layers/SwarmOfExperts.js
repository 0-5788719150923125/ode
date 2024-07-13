import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SwarmOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.hiddenDim = config.hiddenDim || 64
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.hiddenDim, this.numExperts],
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
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Gating network
            const gatingHidden = this.applyDense(
                inputs,
                this.gatingHidden.read(),
                this.gatingHiddenBias.read()
            )
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(gatingHidden)
            const expertWeights = this.applyDense(
                activatedGate,
                this.gatingKernel.read(),
                this.gatingBias.read()
            ).softmax()

            // Compute expert outputs
            const expertOutputs = this.experts.map((expert, i) =>
                this.experts[i].apply(inputs)
            )

            // Combine expert outputs using weighted sum
            const combinedOutput = expertOutputs.reduce((prev, curr, i) => {
                const expertWeight = expertWeights.slice(
                    [0, 0, i],
                    [inputs.shape[0], inputs.shape[1], 1]
                )
                return prev.add(curr.mul(expertWeight))
            }, tf.zeros(expertOutputs[0].shape))

            return combinedOutput
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            activation: this.activation
        }
    }
}

tf.serialization.registerClass(SwarmOfExperts)

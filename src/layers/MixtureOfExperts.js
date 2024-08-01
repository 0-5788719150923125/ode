import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MixtureOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.hiddenDim = config.hiddenDim || 128
        this.activation = config.activation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[0][inputShape[0].length - 1]

        // Initialize gating network
        this.gatingHidden = this.addWeight(
            'gatingHidden',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.gatingHiddenBias = this.addWeight(
            'gatingHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros()
        )
        this.gatingKernel = this.addWeight(
            'gatingKernel',
            [this.hiddenDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.gatingBias = this.addWeight(
            'gatingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros()
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

            // Combine expert outputs using weighted sum
            const combinedOutput = expertInputs.reduce((prev, curr, i) => {
                const expertWeight = expertWeights.slice(
                    [0, 0, i],
                    [inputs.shape[0], inputs.shape[1], 1]
                )
                return prev.add(curr.mul(expertWeight))
            }, tf.zeros(expertInputs[0].shape))

            return combinedOutput
        })
    }

    computeOutputShape(inputShape) {
        return inputShape[0]
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

tf.serialization.registerClass(MixtureOfExperts)

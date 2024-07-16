import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// inspired by SMEAR:
// https://arxiv.org/abs/2306.03745
export default class RoughMergingOfExperts extends LayerBase {
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
        this.routerHiddenKernel = this.addWeight(
            'routerHiddenKernel',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerHiddenBias = this.addWeight(
            'routerHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerOutputKernel = this.addWeight(
            'routerOutputKernel',
            [this.hiddenDim, this.numExperts - 1],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerOutputBias = this.addWeight(
            'routerOutputBias',
            [this.numExperts - 1],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Gating network
            const gatingHidden = this.ops.applyDense(
                inputs,
                this.routerHiddenKernel.read(),
                this.routerHiddenBias.read()
            )

            // Apply layer normalization before activating the logits of our router
            const normalizedState = this.ops.rmsNorm(gatingHidden)
            const activatedGate = tf.layers
                .activation({ activation: this.activation })
                .apply(normalizedState)

            const expertWeights = this.ops.applyDense(
                activatedGate,
                this.routerOutputKernel.read(),
                this.routerOutputBias.read()
            )

            // Merge experts
            const mergedExpert = this.mergeExperts(this.experts, expertWeights)

            // Pass inputs to merged expert
            return mergedExpert.apply(inputs)
        })
    }

    mergeExperts(experts, weights) {
        // We modify the first expert in-place
        const mergedExpert = experts[0]
        // We only use the experts following the first one
        const usedExperts = experts.slice(1)

        for (let i = 0; i < mergedExpert.layers.length; i++) {
            const layer = mergedExpert.layers[i]

            // Compute weighted average of weights for this layer across all experts
            const averagedWeights = layer.getWeights().map((_, weightIndex) => {
                const expertWeights = usedExperts.map(
                    (expert) => expert.layers[i].getWeights()[weightIndex]
                )
                const weightedSum = expertWeights.reduce(
                    (sum, weight, expertIndex) => {
                        const expertWeight = weights
                            .slice([0, 0, expertIndex], [-1, -1, 1])
                            .mean()
                        return sum.add(weight.mul(expertWeight))
                    },
                    tf.zeros(expertWeights[0].shape)
                )
                // Divide by the sum of weights to get the weighted average
                return weightedSum.div(weights.sum())
            })

            // Set the averaged weights to the layer
            layer.setWeights(averagedWeights)
        }

        return mergedExpert
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

tf.serialization.registerClass(RoughMergingOfExperts)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2306.03745
export default class SoftMergingOfExpertsMLP extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.routerDim = config.routerDim || 64
        this.expertDim = config.expertDim || 256
        this.activation = config.activation || 'swish'
        this.routerActivation = config.routerActivation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        // Initialize gating network
        this.routerHiddenKernel = this.addWeight(
            'routerHiddenKernel',
            [inputDim, this.routerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerHiddenBias = this.addWeight(
            'routerHiddenBias',
            [this.routerDim],
            'float32',
            tf.initializers.zeros()
        )
        this.routerOutputKernel = this.addWeight(
            'routerOutputKernel',
            [this.routerDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.routerOutputBias = this.addWeight(
            'routerOutputBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros()
        )

        // Initialize the experts
        this.inProjKernels = []
        this.outProjKernels = []
        for (let i = 0; i < this.numExperts; i++) {
            this.inProjKernels.push(
                this.addWeight(
                    `inProjKernel-${i}`,
                    [inputDim, this.expertDim],
                    'float32',
                    tf.initializers.glorotNormal(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.outProjKernels.push(
                this.addWeight(
                    `outProjKernel-${i}`,
                    [this.expertDim, inputDim],
                    'float32',
                    tf.initializers.glorotNormal(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
        }
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
                .activation({ activation: this.routerActivation })
                .apply(normalizedState)

            const expertWeights = this.ops.applyDense(
                activatedGate,
                this.routerOutputKernel.read(),
                this.routerOutputBias.read()
            )

            const avgInProjKernel = this.computeWeightedAverage(
                this.inProjKernels,
                expertWeights
            )

            const avgOutProjKernel = this.computeWeightedAverage(
                this.outProjKernels,
                expertWeights
            )

            // Expand and contract projection via feedforward layers
            let outputs = this.ops.applyDense(inputs, avgInProjKernel)

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.ops.applyDense(outputs, avgOutProjKernel)

            // Residual connection
            outputs = tf.add(inputs, outputs)

            return outputs
        })
    }

    computeWeightedAverage(tensors, weights) {
        const weightedSum = tensors.reduce((sum, weight, expertIndex) => {
            const expertWeight = weights
                .slice([0, 0, expertIndex], [-1, -1, 1])
                .mean()
            return sum.add(weight.read().mul(expertWeight))
        }, tf.zeros(tensors[0].shape))
        // Divide by the sum of weights to get the weighted average
        return weightedSum.div(weights.sum())
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            routerDim: this.routerDim,
            expertDim: this.expertDim,
            activation: this.activation,
            routerActivation: this.routerActivation
        }
    }
}

tf.serialization.registerClass(SoftMergingOfExpertsMLP)

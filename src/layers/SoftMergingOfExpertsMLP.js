import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

// https://arxiv.org/abs/2306.03745
export default class SoftMergingOfExpertsMLP extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts
        this.routerDim = config.routerDim || 64
        this.expertDim = config.expertDim || 256
        this.activation = config.activation || 'swish'
        this.routerActivation = config.routerActivation || 'swish'
        this.gateActivation = config.gateActivation || 'swish'
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.routerHiddenKernel = this.addWeight(
            'routerHiddenKernel',
            [inputDim, this.routerDim],
            'float32',
            this.initializers.glorotNormal()
        )
        this.routerOutputKernel = this.addWeight(
            'routerOutputKernel',
            [this.routerDim, this.numExperts],
            'float32',
            this.initializers.glorotNormal()
        )
        if (this.useBias) {
            this.routerHiddenBias = this.addWeight(
                'routerHiddenBias',
                [this.routerDim],
                'float32',
                this.initializers.zeros()
            )
            this.routerOutputBias = this.addWeight(
                'routerOutputBias',
                [this.numExperts],
                'float32',
                this.initializers.zeros()
            )
        }

        this.inProjKernels = []
        this.gateProjKernels = []
        this.outProjKernels = []
        this.inProjBiases = []
        this.gateProjBiases = []
        this.outProjBiases = []
        for (let i = 0; i < this.numExperts; i++) {
            this.inProjKernels.push(
                this.addWeight(
                    `inProjKernel-${i}`,
                    [inputDim, this.expertDim],
                    'float32',
                    this.initializers.glorotNormal()
                )
            )
            this.gateProjKernels.push(
                this.addWeight(
                    `gateProjKernel-${i}`,
                    [inputDim, this.expertDim],
                    'float32',
                    this.initializers.glorotNormal()
                )
            )
            this.outProjKernels.push(
                this.addWeight(
                    `outProjKernel-${i}`,
                    [this.expertDim, inputDim],
                    'float32',
                    this.initializers.glorotNormal()
                )
            )
            if (this.useBias) {
                this.inProjBiases.push(
                    this.addWeight(
                        `inProjBias-${i}`,
                        [this.expertDim],
                        'float32',
                        this.initializers.zeros()
                    )
                )
                this.gateProjBiases.push(
                    this.addWeight(
                        `gateProjBias-${i}`,
                        [this.expertDim],
                        'float32',
                        this.initializers.zeros()
                    )
                )
                this.outProjBiases.push(
                    this.addWeight(
                        `outProjBias-${i}`,
                        [inputDim],
                        'float32',
                        this.initializers.zeros()
                    )
                )
            }
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const gatingHidden = this.ops.applyDense(
                inputs,
                this.routerHiddenKernel.read(),
                this.routerHiddenBias?.read()
            )

            const activatedGate = tf.layers
                .activation({ activation: this.routerActivation })
                .apply(gatingHidden)

            const expertWeights = this.ops.applyDense(
                activatedGate,
                this.routerOutputKernel.read(),
                this.routerOutputBias?.read()
            )

            const avgInProjKernel = this.computeWeightedAverage(
                this.inProjKernels,
                expertWeights
            )

            const avgGateProjKernel = this.computeWeightedAverage(
                this.gateProjKernels,
                expertWeights
            )

            const avgOutProjKernel = this.computeWeightedAverage(
                this.outProjKernels,
                expertWeights
            )

            let avgInProjBias = null
            let avgGateProjBias = null
            let avgOutProjBias = null
            if (this.useBias) {
                avgInProjBias = this.computeWeightedAverage(
                    this.inProjBiases,
                    expertWeights
                )
                avgGateProjBias = this.computeWeightedAverage(
                    this.gateProjBiases,
                    expertWeights
                )
                avgOutProjBias = this.computeWeightedAverage(
                    this.outProjBiases,
                    expertWeights
                )
            }

            let proj = this.ops.applyDense(
                inputs,
                avgInProjKernel,
                avgInProjBias
            )

            proj = tf.layers
                .activation({ activation: this.activation })
                .apply(proj)

            let gate = this.ops.applyDense(
                inputs,
                avgGateProjKernel,
                avgGateProjBias
            )

            gate = tf.layers
                .activation({ activation: this.gateActivation })
                .apply(gate)

            const gatedOutput = tf.mul(proj, gate)

            let outputs = this.ops.applyDense(
                gatedOutput,
                avgOutProjKernel,
                avgOutProjBias
            )

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
            routerActivation: this.routerActivation,
            gateActivation: this.gateActivation
        }
    }
}

tf.serialization.registerClass(SoftMergingOfExpertsMLP)

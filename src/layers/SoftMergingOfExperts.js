import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2306.03745
export default class SoftMergingOfExperts extends LayerBase {
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
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        this.routerHiddenBias = this.addWeight(
            'routerHiddenBias',
            [this.hiddenDim],
            'float32',
            tf.initializers.zeros()
        )
        this.routerOutputKernel = this.addWeight(
            'routerOutputKernel',
            [this.hiddenDim, this.numExperts - 1],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        this.routerOutputBias = this.addWeight(
            'routerOutputBias',
            [this.numExperts - 1],
            'float32',
            tf.initializers.zeros()
        )

        for (const expert of this.experts) {
            expert.build(inputShape)
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

        const collectedWeights = this.collectExpertWeights(usedExperts)

        for (const i in collectedWeights) {
            const averagedWeights = this.computeWeightedAverage(
                collectedWeights[i],
                weights
            )
            const variables = this.findLayerVariables(experts[0].layers[1])
            mergedExpert.layers[1][variables[i]].write(averagedWeights)
        }

        return mergedExpert
    }

    collectExpertWeights(experts) {
        const allWeights = []
        for (let i = 0; i < experts[0].layers.length; i++) {
            const variables = this.findLayerVariables(experts[0].layers[i])
            for (const v of variables) {
                const parallelWeights = []
                for (const expert of experts) {
                    parallelWeights.push(expert.layers[i][v].read())
                }
                allWeights.push(parallelWeights)
            }
        }
        return allWeights
    }

    findLayerVariables(layer) {
        const layerVariables = []

        for (const key in layer) {
            if (layer.hasOwnProperty(key)) {
                const value = layer[key]
                if (value instanceof tf.LayerVariable) {
                    layerVariables.push(key)
                }
            }
        }

        return layerVariables
    }

    computeWeightedAverage(tensors, weights) {
        const weightedSum = tensors.reduce((sum, weight, expertIndex) => {
            const expertWeight = weights
                .slice([0, 0, expertIndex], [-1, -1, 1])
                .mean()
            return sum.add(weight.mul(expertWeight))
        }, tf.zeros(tensors[0].shape))
        // Divide by the sum of weights to get the weighted average
        return weightedSum.div(weights.sum())
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

tf.serialization.registerClass(SoftMergingOfExperts)

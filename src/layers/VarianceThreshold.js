import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class VarianceThreshold extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
    }

    build(inputShape) {
        this.inputDim = inputShape[2]
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const selectedFeatures = this.applyVarianceThreshold(inputs)
            return selectedFeatures
        })
    }

    applyVarianceThreshold(inputs) {
        const batchSize = inputs.shape[0]
        const timeSteps = inputs.shape[1]
        const inputDim = inputs.shape[2]

        // Reshape the input tensor to [batchSize * timeSteps, inputDim]
        const flattenedInputs = inputs.reshape([
            batchSize * timeSteps,
            inputDim
        ])

        // Calculate the variance of each feature
        const variance = tf.moments(flattenedInputs, 0).variance

        // Sort the variances in descending order
        const sortedVariances = variance
            .arraySync()
            .map((v, i) => [v, i])
            .sort((a, b) => b[0] - a[0])

        // Get the indices of the top outputDim features
        const selectedIndices = sortedVariances
            .slice(0, this.outputDim)
            .map((x) => x[1])

        // Create a boolean mask based on the selected indices
        const mask = tf
            .zeros([this.inputDim])
            .arraySync()
            .map((_, i) => selectedIndices.includes(i))

        // Apply the mask to select the features
        const selectedFlattenedFeatures = flattenedInputs.mul(mask)

        // Reshape the selected features back to [batchSize, timeSteps, outputDim]
        const selectedFeatures = selectedFlattenedFeatures.reshape([
            batchSize,
            timeSteps,
            this.outputDim
        ])

        return selectedFeatures
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(VarianceThreshold)

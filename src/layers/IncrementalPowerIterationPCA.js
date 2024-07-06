import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class IncrementalPowerIterationPCA extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
        this.epsilon = config.epsilon || 1e-7
        this.numIterations = config.numIterations || 10
        this.mean = null
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.inputDim = inputDim
        this.components = this.addWeight(
            'components',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers.glorotNormal({})
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            this.fit(inputs)

            // Center the data
            const centered = tf.sub(inputs, this.mean)

            // Project data onto principal components
            const flattenedCentered = tf.reshape(centered, [-1, this.inputDim])
            const components = this.components.read()

            const result = tf.matMul(flattenedCentered, components)
            const reshapedResult = tf.reshape(result, [
                ...inputs.shape.slice(0, -1),
                this.outputDim
            ])

            return reshapedResult
        })
    }

    fit(X) {
        tf.tidy(() => {
            const flattenedX = tf.reshape(X, [-1, this.inputDim])

            // Compute mean
            this.mean = tf.mean(flattenedX, 0, true)

            // Center the data
            const centered = tf.sub(flattenedX, this.mean)

            // Compute covariance matrix
            const cov = tf
                .matMul(centered, centered, true, false)
                .div(tf.scalar(centered.shape[0] - 1))

            // Compute principal components using power iteration
            const components = this.powerIteration(cov)
        })
    }

    powerIteration(cov) {
        return tf.tidy(() => {
            const [n] = cov.shape
            let components = tf.randomNormal([n, this.outputDim])

            for (let iter = 0; iter < this.numIterations; iter++) {
                // Power iteration
                components = tf.matMul(cov, components)

                // Orthogonalize
                components = tf.linalg.gramSchmidt(components)
            }

            return components
        })
    }

    computeOutputShape(inputShape) {
        return [...inputShape.slice(0, -1), this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            epsilon: this.epsilon,
            numIterations: this.numIterations
        }
    }
}

tf.serialization.registerClass(IncrementalPowerIterationPCA)

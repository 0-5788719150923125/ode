import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'
import customActivations from '../activations.js'

export default class ParabolicCompression extends LayerBase {
    constructor(config) {
        super(config)
        this.numSteps = config.numSteps || 3
        this.outputDim = config.outputDim || 64
        this.activation = customActivations.Snake
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.stepSize = (inputDim - this.outputDim) / this.numSteps

        if (inputDim % this.stepSize !== 0) {
            throw `inputDim (${inputDim}) must be a multiple of stepSize (${this.stepSize})!`
        }

        this.alpha = []
        this.beta = []
        this.gamma = []
        this.projectionMatrices = []

        let currentSize = inputDim

        for (let i = 0; i < this.numSteps; i++) {
            const newSize = inputDim - this.stepSize * (i + 1)

            if (newSize < this.outputDim) {
                throw `newSize (${newSize}) should never be smaller than outputDim ${this.outputDim}!`
            }

            this.projectionMatrices.push(
                this.addWeight(
                    `projection-${i}`,
                    [currentSize, newSize],
                    'float32',
                    tf.initializers.orthogonal({ gain: 1.0 })
                )
            )

            this.alpha.push(
                this.addWeight(
                    `alpha-${i}`,
                    [1, newSize],
                    'float32',
                    tf.initializers.ones()
                )
            )
            this.beta.push(
                this.addWeight(
                    `beta-${i}`,
                    [1, newSize],
                    'float32',
                    tf.initializers.zeros()
                )
            )
            this.gamma.push(
                this.addWeight(
                    `gamma-${i}`,
                    [1, newSize],
                    'float32',
                    tf.initializers.zeros()
                )
            )

            currentSize = newSize
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            let outputs = inputs

            for (let i = 0; i < this.numSteps; i++) {
                // Slice the weights based on the input dimensions
                // const inProj = matrix.slice([0, 0], [-1, newSize])
                // const outProj = matrix.slice([0, inputDim - newSize], [-1, -1])
                outputs = this.ops.applyDense(
                    outputs,
                    this.projectionMatrices[i].read()
                )

                const newSize = outputs.shape[outputs.shape.length - 1]

                // Reshape activation parameters to match outputs
                const alpha = tf.reshape(this.alpha[i].read(), [1, 1, newSize])
                const beta = tf.reshape(this.beta[i].read(), [1, 1, newSize])
                const gamma = tf.reshape(this.gamma[i].read(), [1, 1, newSize])

                // Per-neuron activation via Snake
                outputs = this.activation.apply(outputs, alpha, beta, gamma)
            }

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numSteps: this.numSteps,
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(ParabolicCompression)

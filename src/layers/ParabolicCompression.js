import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'
import customActivations from '../activations.js'

export default class ParabolicCompression extends LayerBase {
    constructor(config) {
        super(config)
        this.numSteps = config.numSteps || 3
        this.units = config.units || 64
        this.activation = customActivations.Snake
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.stepSize = (inputDim - this.units) / this.numSteps

        if ((inputDim - this.units) % this.stepSize !== 0) {
            throw `inputDim (${inputDim}) minus units (${this.units}) must be a multiple of stepSize (${this.stepSize})!`
        }

        this.alpha = []
        this.beta = []
        this.gamma = []

        this.projectionMatrix = this.addWeight(
            `projectionMatrix`,
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.residualMatrix = this.addWeight(
            'residualMatrix',
            [inputDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )

        for (let i = 0; i < this.numSteps; i++) {
            const newSize = inputDim - this.stepSize * (i + 1)

            if (newSize < this.units) {
                throw `newSize (${newSize}) should never be smaller than units ${this.units}!`
            }

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
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLength, inputDim] = inputs.shape

            let outputs = inputs

            const matrix = this.projectionMatrix.read()

            let currentSize = inputDim
            for (let i = 0; i < this.numSteps; i++) {
                const newSize = inputDim - this.stepSize * (i + 1)

                // Project inputs into a lower dimension
                const inProj = matrix.slice([0, 0], [currentSize, newSize])
                outputs = this.ops.applyDense(outputs, inProj)

                // Reshape activation parameters to match outputs
                const alpha = tf.reshape(this.alpha[i].read(), [1, 1, newSize])
                const beta = tf.reshape(this.beta[i].read(), [1, 1, newSize])
                const gamma = tf.reshape(this.gamma[i].read(), [1, 1, newSize])

                // Per-feature activation via Snake
                outputs = this.activation.apply(outputs, alpha, beta, gamma)

                // Reduce the size
                currentSize = newSize

                // Project outputs via a smaller, shifted slice of the matrix
                const outProj = matrix.slice(
                    [inputDim - currentSize, inputDim - newSize],
                    [-1, -1]
                )
                outputs = this.ops.applyDense(outputs, outProj)
            }

            const inputReshaped = inputs.reshape([-1, inputDim])
            const residualOutput = inputReshaped
                .matMul(this.residualMatrix.read())
                .reshape([batchSize, seqLength, this.units])

            return outputs.add(residualOutput)
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numSteps: this.numSteps,
            units: this.units
        }
    }
}

tf.serialization.registerClass(ParabolicCompression)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'
import customActivations from '../activations.js'

export default class ParabolicCompression extends LayerBase {
    constructor(config) {
        super(config)
        this.numSteps = config.numSteps || 3
        this.outputDim = config.outputDim || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.stepSize = (inputDim - this.outputDim) / this.numSteps
        if (inputDim % this.stepSize !== 0) {
            throw `inputDim (${inputDim}) must be a multiple of stepSize (${this.stepSize})!`
        }

        this.activation = customActivations.Snake
        this.transformationMatrix = this.addWeight(
            'transformationMatrix',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.alpha = []
        this.beta = []
        this.gamma = []
        for (let i = 0; i < this.numSteps; i++) {
            this.alpha.push(
                this.addWeight(
                    `alpha-${i}`,
                    [],
                    'float32',
                    tf.initializers.ones()
                )
            )
            this.beta.push(
                this.addWeight(
                    `beta-${i}`,
                    [],
                    'float32',
                    tf.initializers.zeros()
                )
            )
            this.gamma.push(
                this.addWeight(
                    `gamma-${i}`,
                    [],
                    'float32',
                    tf.initializers.zeros()
                )
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Get the input dimensions
            const inputDim = inputs.shape[inputs.shape.length - 1]

            // const stepSize = inputDim / this.numSteps

            let outputs = inputs

            const matrix = this.transformationMatrix.read()
            for (let i = 0; i < this.numSteps; i++) {
                const newSize = inputDim - this.stepSize * (i + 1)
                // Slice the weights based on the input dimensions
                const inProj = matrix.slice([0, 0], [-1, newSize])
                const outProj = matrix.slice([0, inputDim - newSize], [-1, -1])
                outputs = this.ops.applyDense(outputs, inProj)
                outputs = this.activation.apply(
                    outputs,
                    this.alpha[i].read(),
                    this.beta[i].read(),
                    this.gamma[i].read()
                )
                outputs = this.ops.applyDense(outputs, tf.transpose(outProj))
            }

            return outputs
        })
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

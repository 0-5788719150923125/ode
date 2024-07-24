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
        this.activation = customActivations.Snake
        this.transformer = this.addWeight(
            'transformer',
            [inputDim, this.rank],
            'float32',
            tf.initializers.glorotNormal()
        )
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Get the input dimensions
            const inputDim = inputs.shape[inputs.shape.length - 1]

            // Slice the weights based on the input dimensions
            const slicedInProjKernel = this.inProjKernel
                .read()
                .slice([0, 0], [inputDim, this.innerDim])
            const slicedOutProjKernel = this.outProjKernel
                .read()
                .slice([0, 0], [this.innerDim, inputDim])
            const slicedOutProjBias = this.outProjBias
                .read()
                .slice([0], [inputDim])

            return this.activation.apply(inputs, alpha, beta, gamma)
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

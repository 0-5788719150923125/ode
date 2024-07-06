import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureReduction extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
        this.scale = config.scale || 1.0
        this.seed = config.seed || 42
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.projectionMatrix = tf.randomNormal(
            [this.inputDim, this.outputDim],
            0,
            this.scale / Math.sqrt(this.outputDim),
            'float32',
            this.seed
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLength, inputDim] = input.shape

            // Reshape to 2D for matrix multiplication
            const inputReshaped = input.reshape([-1, inputDim])

            // Apply random projection
            const projected = tf.matMul(inputReshaped, this.projectionMatrix)

            // Reshape back to 3D
            return projected.reshape([batchSize, seqLength, this.outputDim])
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            scale: this.scale,
            seed: this.seed
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureReduction)

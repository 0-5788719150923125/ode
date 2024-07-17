import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
export default class ProjectedFeatureReduction extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
        this.scale = config.scale || 1.0
        this.seed = config.seed || null
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.projectionMatrix = this.addWeight(
            'projectionMatrix',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers.randomNormal({
                mean: 0,
                stdDev: this.scale / Math.sqrt(this.outputDim),
                seed: this.seed
            }),
            null,
            false
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLength, inputDim] = input.shape

            // Reshape to 2D for matrix multiplication
            const inputReshaped = input.reshape([-1, inputDim])

            // Apply random projection
            const projected = tf.matMul(
                inputReshaped,
                this.projectionMatrix.read()
            )

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

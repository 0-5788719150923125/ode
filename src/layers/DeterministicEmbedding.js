import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class DeterministicEmbedding extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
    }

    computeOutputShape(inputShape) {
        return [...inputShape, this.outputDim]
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const tokenIds = inputs.cast('int32')
            const positions = tf
                .range(0, inputs.shape[1])
                .expandDims(0)
                .cast('int32')

            const tokenEncodings = tf
                .oneHot(tokenIds, this.outputDim)
                .cast('float32')
            const positionEncodings = tf
                .oneHot(positions, this.outputDim)
                .cast('float32')

            const encodings = tokenEncodings.add(positionEncodings)
            const normalizedEncodings = encodings.div(
                tf.sqrt(tf.scalar(this.outputDim))
            )

            return normalizedEncodings
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(DeterministicEmbedding)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class RotaryPositionalEncoding extends LayerBase {
    constructor(config) {
        super(config)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]
            const embeddingDim = inputs.shape[2]
            const posEncoding = this.getRotaryPositionalEmbedding(
                seqLen,
                embeddingDim
            )
            const output = this.applyRotaryPositionalEmbedding(
                inputs,
                posEncoding
            )
            return output.slice([0, 0, 0], inputs.shape)
        })
    }

    getRotaryPositionalEmbedding(seqLen, embeddingDim) {
        const pos = tf.range(0, seqLen, 1, 'float32').reshape([-1, 1])
        const i = tf.range(0, embeddingDim / 2, 1, 'float32').reshape([1, -1])
        const angleRates = tf.pow(10000, tf.div(i, embeddingDim / 2))
        const angleRads = tf.mul(pos, tf.div(1, angleRates))
        const sin = tf.sin(angleRads)
        const cos = tf.cos(angleRads)

        // Interleave sin and cos values
        const sinExpanded = sin.expandDims(2) // Expanding dimension to enable interleaving
        const cosExpanded = cos.expandDims(2)
        const concatenated = tf.concat([sinExpanded, cosExpanded], 2) // Concatenate along the new axis
        const posEncoding = concatenated.reshape([seqLen, embeddingDim])
        return posEncoding
    }

    applyRotaryPositionalEmbedding(x, posEncoding) {
        const embeddingDim = x.shape[2]
        const xDtype = x.dtype

        // Split the embedding dimension into two halves for sin and cos applications
        const rotaryDim = embeddingDim / 2
        const [xRot, xPass] = tf.split(x, 2, -1)

        // Apply sin to the first half and cos to the second half of posEncoding
        const sinEncoding = posEncoding.slice([0, 0], [-1, rotaryDim])
        const cosEncoding = posEncoding.slice([0, rotaryDim], [-1, -1])

        // Apply the encodings
        const xRotSin = tf.mul(xRot, sinEncoding)
        const xRotCos = tf.mul(xRot, cosEncoding)

        // Reconstruct the rotated embeddings
        const rotatedX = tf.concat([xRotSin, xRotCos], -1)

        // Concatenate the rotated part with the part that does not get rotated
        const output = tf.concat([rotatedX, xPass], -1)

        return output.asType(xDtype)
    }

    getConfig() {
        return {
            ...super.getConfig()
        }
    }
}

tf.serialization.registerClass(RotaryPositionalEncoding)

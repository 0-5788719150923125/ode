import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class SharedEmbedding extends LayerBase {
    constructor(config) {
        super(config)
        this.inputDim = config.inputDim
        this.outputDim = config.outputDim
        this.embeddingsInitializer =
            config.embeddingsInitializer || 'glorotUniform'
        this.dropout = config.dropout || 0
    }

    build(inputShape) {
        this.embeddings = this.addWeight(
            'embeddings',
            [this.inputDim, this.outputDim],
            'float32',
            tf.initializers[this.embeddingsInitializer]({
                seed: this.ops.getSeed()
            })
        )
    }

    computeOutputShape(inputShape) {
        if (inputShape.length === 2) {
            // Input embedding
            return [inputShape[0], inputShape[1], this.outputDim]
        } else if (inputShape.length === 3) {
            // Output projection
            return [inputShape[0], inputShape[1], this.inputDim]
        } else {
            throw new Error('Invalid input shape for TiedEmbedding layer.')
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            if (inputs.shape.length === 2) {
                // Input embedding
                const flatInputs = tf.reshape(inputs, [-1])
                const embeddings = tf.gather(
                    this.embeddings.read(),
                    flatInputs.cast('int32')
                )

                let outputs = tf.reshape(embeddings, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.outputDim
                ])

                outputs = kwargs['training']
                    ? tf.dropout(outputs, this.dropout)
                    : outputs

                return outputs
            } else if (inputs.shape.length === 3) {
                // Output projection
                const denseOutput = tf.matMul(
                    tf.reshape(inputs, [-1, this.outputDim]),
                    this.embeddings.read(),
                    false,
                    true
                )

                let outputs = tf.reshape(denseOutput, [
                    inputs.shape[0],
                    inputs.shape[1],
                    this.inputDim
                ])

                outputs = kwargs['training']
                    ? tf.dropout(outputs, this.dropout)
                    : outputs

                return outputs
            } else {
                throw new Error(
                    'Invalid input shape for SharedEmbedding layer.'
                )
            }
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            inputDim: this.inputDim,
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(SharedEmbedding)

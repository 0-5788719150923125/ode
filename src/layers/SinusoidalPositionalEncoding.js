import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'
import Range from './Range.js'

export default class SinusoidalPositionalEncoding extends LayerBase {
    constructor(config) {
        super(config)
        this.reverse = config?.reverse || false
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const range = new Range().apply(inputs)

            // Determine the sequence length from the input shape
            const seqLength = range.shape[1]

            // Compute the positional encodings (2D tensor of shape [seqLength, this.units])
            const positionalEncoding = tf.tensor2d(
                Array.from({ length: seqLength }, (_, pos) => {
                    return Array.from({ length: inputs.shape[2] }, (_, i) => {
                        const divTerm = Math.pow(
                            10000,
                            (2 * (i / 2)) / inputs.shape[2]
                        )
                        // Switch between sine and cosine based on the flag
                        if (this.reverse) {
                            return i % 2 === 0
                                ? Math.cos(pos / divTerm)
                                : Math.sin(pos / divTerm)
                        } else {
                            return i % 2 === 0
                                ? Math.sin(pos / divTerm)
                                : Math.cos(pos / divTerm)
                        }
                    })
                })
            )

            // Broadcast the positional encoding to match the shape of the inputs
            const broadcastedPositionalEncoding = positionalEncoding
                .expandDims(0)
                .tile([inputs.shape[0], 1, 1])

            return inputs.add(broadcastedPositionalEncoding)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }
}

tf.serialization.registerClass(SinusoidalPositionalEncoding)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class Antirectifier extends LayerBase {
    constructor(config) {
        super(config)
        // TODO(bileschi): Can we point to documentation on masking here?
        this.supportsMasking = true
    }

    /**
     * This layer only works on 4D Tensors [batch, height, width, channels],
     * and produces output with twice as many channels.
     *
     * layer.computeOutputShapes must be overridden in the case that the output
     * shape is not the same as the input shape.
     * @param {*} inputShapes
     */
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], inputShape[2], 2 * inputShape[3]]
    }

    /**
     * Centers the input and applies the following function to every element of
     * the input.
     *
     *     x => [max(x, 0), max(-x, 0)]
     *
     * The theory being that there may be signal in the both negative and positive
     * portions of the input.  Note that this will double the number of channels.
     * @param inputs Tensor to be treated.
     * @param kwargs Only used as a pass through to call hooks.  Unused in this
     *   example code.
     */
    call(inputs, kwargs) {
        let input = inputs
        if (Array.isArray(input)) {
            input = input[0]
        }

        const origShape = input.shape
        const flatShape = [
            origShape[0],
            origShape[1] * origShape[2] * origShape[3]
        ]
        const flattened = input.reshape(flatShape)
        const centered = tf.sub(flattened, flattened.mean(1).expandDims(1))
        const pos = centered.relu().reshape(origShape)
        const neg = centered.neg().relu().reshape(origShape)
        return tf.concat([pos, neg], 3)
    }
}

tf.serialization.registerClass(Antirectifier)

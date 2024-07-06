import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class Range extends LayerBase {
    constructor(config) {
        super(config)
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const [B, T] = input.shape
            const range = tf.reshape(tf.range(0, T, 1, 'int32'), [1, T]) // .tile([B, 1])
            return range
        })
    }
}

tf.serialization.registerClass(Range)

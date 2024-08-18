import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class Zip extends LayerBase {
    constructor(config) {
        super(config)
    }

    computeOutputShape(inputShape) {
        const shape1 = inputShape[0]
        const shape2 = inputShape[1]
        const outputShape = shape1.slice()
        outputShape[2] = shape1[2] + shape2[2]
        return outputShape
    }

    call(inputs, kwargs) {
        return this.ops.zip(inputs[0], inputs[1])
    }
}

tf.serialization.registerClass(Zip)

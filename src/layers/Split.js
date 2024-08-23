import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class Split extends LayerBase {
    constructor(config) {
        super(config)
        this.axis = config.axis || -1
        this.numOrSizeSplits = config.numOrSizeSplits || 2
    }

    computeOutputShape(inputShape) {
        const outputShape = inputShape.slice()
        const axis = this.axis < 0 ? inputShape.length + this.axis : this.axis

        if (Array.isArray(this.numOrSizeSplits)) {
            return this.numOrSizeSplits.map((size) => {
                const shape = outputShape.slice()
                shape[axis] = size
                return shape
            })
        } else {
            const splitSize = Math.floor(
                inputShape[axis] / this.numOrSizeSplits
            )
            return Array(this.numOrSizeSplits)
                .fill()
                .map(() => {
                    const shape = outputShape.slice()
                    shape[axis] = splitSize
                    return shape
                })
        }
    }

    call(inputs, kwargs) {
        return tf.split(inputs[0], this.numOrSizeSplits, this.axis)
    }

    getConfig() {
        return {
            ...super.getConfig(),
            axis: this.axis,
            numOrSizeSplits: this.numOrSizeSplits
        }
    }
}

tf.serialization.registerClass(Split)

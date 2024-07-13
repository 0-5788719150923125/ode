import * as tf from '@tensorflow/tfjs'
import customOps from '../ops.js'

export default class LayerBase extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.ops = customOps
        this.useALiBi = config.useALiBi || false
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // stuff should go here
        })
    }

    static get className() {
        return this.name
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            className: this.getClassName(),
            useALiBi: this.useALiBi
        }
    }
}

import * as tf from '@tensorflow/tfjs'
import customOps from '../ops.js'
import customInitializers from '../initializers.js'

export default class LayerBase extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.ops = customOps
        this.initializers = customInitializers
        this.useBias = config?.useBias || false
        this.ALiBiLength = config?.ALiBiLength || false
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
            useBias: this.useBias,
            ALiBiLength: this.ALiBiLength
        }
    }
}

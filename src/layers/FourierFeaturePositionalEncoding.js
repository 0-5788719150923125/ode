import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class FourierFeaturePositionalEncoding extends LayerBase {
    constructor(config) {
        super(config)
        this.numFeatures = config.numFeatures
        this.scale = config.scale || 1.0
    }

    call(inputs) {
        const position = tf.range(0, inputs.shape[1], 1, 'float32')
        const angles = tf.mul(
            position,
            tf.pow(10000.0, tf.linspace(0.0, 1.0, this.numFeatures))
        )
        const cosFeatures = tf.cos(angles)
        const sinFeatures = tf.sin(angles)
        const features = tf.stack([cosFeatures, sinFeatures], -1)
        const flattened = tf.reshape(features, [1, -1, this.numFeatures * 2])
        const scaled = tf.mul(flattened, this.scale)
        return tf.add(inputs, scaled)
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            numFeatures: this.numFeatures,
            scale: this.scale
        })
        return config
    }
}

tf.serialization.registerClass(FourierFeaturePositionalEncoding)

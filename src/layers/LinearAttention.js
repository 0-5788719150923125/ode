import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class LinearAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units || 64
        this.projection = config.projection || 256
        this.numFeatures = config.numFeatures || 256
    }

    build(inputShape) {
        this.query = tf.layers.dense({
            units: this.projection,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
        this.key = tf.layers.dense({
            units: this.projection,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
        this.value = tf.layers.dense({
            units: this.units,
            activation: 'linear',
            useBias: false,
            kernelInitializer: tf.initializers.glorotUniform()
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.query.apply(inputs)
            const K = this.key.apply(inputs)
            const V = this.value.apply(inputs)

            const Qp = this.generateRandomFeatures(Q)
            const Kp = this.generateRandomFeatures(K)

            const scores = tf.matMul(Qp, Kp, false, true)

            const weights = scores.div(tf.scalar(this.numFeatures))

            const outputs = tf.matMul(weights, V)

            return tf.add(inputs, outputs)
        })
    }

    generateRandomFeatures(inputs) {
        const dims = inputs.shape[inputs.shape.length - 1]
        const W = tf.randomNormal([dims, this.numFeatures])
        const b = tf.randomUniform([this.numFeatures], 0, 2 * Math.PI)
        const features = tf.matMul(inputs, W).add(b).cos()
        return features
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            projection: this.projection,
            numFeatures: this.numFeatures
        }
    }
}

tf.serialization.registerClass(LinearAttention)

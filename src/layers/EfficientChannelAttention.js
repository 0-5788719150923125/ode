import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/1709.01507
export default class EfficientChannelAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.gamma = config.gamma || 2
    }

    build(inputShape) {
        this.channels = inputShape[inputShape.length - 1]
        this.kernelSize = Math.max(1, Math.floor(this.channels / this.gamma))

        this.conv1d = tf.layers.conv1d({
            filters: 1,
            kernelSize: this.kernelSize,
            strides: 1,
            padding: 'same',
            activation: 'sigmoid',
            kernelInitializer: 'ones',
            useBias: false
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const avgPool = tf.mean(inputs, [1], true)
            const attention = this.conv1d.apply(avgPool)

            return inputs.mul(attention)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            gamma: this.gamma
        }
    }
}

tf.serialization.registerClass(EfficientChannelAttention)

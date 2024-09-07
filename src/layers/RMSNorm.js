import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class RMSNorm extends LayerBase {
    constructor(config) {
        super(config)
        this.epsilon = config.epsilon || 1e-6
        this.axis = config.axis || 2
        this.elementwiseAffine = config.elementwiseAffine || true
    }

    build(inputShape) {
        const axisShape = inputShape[this.axis]

        if (this.elementwiseAffine) {
            this.rmsWeight = this.addWeight(
                'weight',
                [axisShape],
                'float32',
                tf.initializers.ones()
            )

            if (this.useBias) {
                this.rmsBias = this.addWeight(
                    'bias',
                    [axisShape],
                    'float32',
                    tf.initializers.zeros()
                )
            }
        }
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const variance = tf.mean(tf.square(inputs), this.axis, true)
            const normalized = tf.mul(
                inputs,
                tf.rsqrt(tf.add(variance, this.epsilon))
            )

            if (this.elementwiseAffine) {
                let output = tf.mul(normalized, this.rmsWeight.read())
                if (this.useBias) {
                    output = tf.add(output, this.rmsBias.read())
                }
                return output
            }

            return normalized
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            epsilon: this.epsilon,
            axis: this.axis,
            elementwiseAffine: this.elementwiseAffine,
            useBias: this.useBias
        }
    }
}

tf.serialization.registerClass(RMSNorm)

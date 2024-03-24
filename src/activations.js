import tf from '@tensorflow/tfjs'

export class GELU extends tf.layers.Layer {
    constructor() {
        super({})
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
            const cdf = tf.mul(
                0.5,
                tf.add(
                    1,
                    tf.tanh(
                        tf.mul(
                            tf.sqrt(tf.div(2, Math.PI)),
                            tf.add(input, tf.mul(0.044715, tf.pow(input, 3)))
                        )
                    )
                )
            )
            return tf.mul(input, cdf)
        })
    }

    static get className() {
        return 'GELU'
    }
}

tf.serialization.registerClass(GELU)

import * as tf from '@tensorflow/tfjs'

/**
 * Gelu activation function
 */
class Gelu extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'gelu'
    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x) {
        return tf.tidy(() => {
            const sqrtTwo = Math.sqrt(2)
            // Compute Φ(x) using the erf function
            const cdf = tf.mul(0.5, tf.add(1, tf.erf(tf.div(x, sqrtTwo))))
            // Compute GELU(x) = x * Φ(x)
            return tf.mul(x, cdf)
        })
    }
}

tf.serialization.registerClass(Gelu)

/**
 * GeluNew activation function
 */
export class GeluNew extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'gelu_new'
    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x) {
        return tf.tidy(() => {
            return tf.mul(
                0.5,
                tf.mul(
                    x,
                    tf.add(
                        1,
                        tf.tanh(
                            tf.mul(
                                tf.sqrt(tf.div(2, Math.PI)),
                                tf.add(x, tf.mul(0.044715, tf.pow(x, 3)))
                            )
                        )
                    )
                )
            )
        })
    }
}

tf.serialization.registerClass(GeluNew)

/**
 * APTx activation function
 */
class APTx extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'aptx'

    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x, alpha = 1.0, beta = 1.0, gamma = 0.5) {
        return tf.tidy(() => {
            const tanhTerm = tf.tanh(tf.mul(beta, x))
            const modulation = tf.add(alpha, tanhTerm)
            return tf.mul(tf.mul(gamma, x), modulation)
        })
    }
}

tf.serialization.registerClass(APTx)

/**
 * Snake activation function
 */
class Snake extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'snake'

    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x, alpha = 1.0) {
        return tf.tidy(() => {
            return tf.add(x, tf.div(tf.square(tf.sin(tf.mul(alpha, x))), alpha))
        })
    }
}

tf.serialization.registerClass(Snake)

const activations = {
    Gelu: new Gelu(),
    GeluNew: new GeluNew(),
    APTx: new APTx(),
    Snake: new Snake()
}

export default activations

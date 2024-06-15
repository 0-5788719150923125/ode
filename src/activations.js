import * as tf from '@tensorflow/tfjs'

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
    APTx: new APTx(),
    Snake: new Snake()
}

export default activations

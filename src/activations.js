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

const activations = {
    Gelu: new Gelu(),
    APTx: (epsilon, omega) => new APTx(epsilon, omega)
}

export default activations

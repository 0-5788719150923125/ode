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
     * @param alpha Scalar value for the alpha parameter.
     * @returns a Tensor of the same shape as x
     */
    apply(x, alpha = 1.0, beta = 0, gamma = 0, epsilon = 1e-8) {
        return tf.tidy(() => {
            // Original Snake activation
            const sinTerm = tf.sin(tf.mul(alpha, x))
            const squaredSinTerm = tf.square(sinTerm)
            const inverseAlpha = tf.scalar(1).div(alpha)
            const snakeTerm = tf.mul(inverseAlpha, squaredSinTerm)

            // Additional oscillation terms
            const outerOscillation = tf.mul(
                tf.add(gamma, epsilon),
                tf.sin(tf.mul(tf.add(beta, epsilon), x))
            )

            // Combine terms
            return tf.add(tf.add(x, snakeTerm), outerOscillation)
        })
    }
}

tf.serialization.registerClass(Snake)

/**
 * Laplace activation function
 */
class LaplaceActivation extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'laplace'

    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @param mu Mean value (default: 0.707107).
     * @param sigma Standard deviation (default: 0.282095).
     * @returns a Tensor of the same shape as x
     */
    apply(x, mu = 0.707107, sigma = 0.282095) {
        return tf.tidy(() => {
            const adjustedInput = tf.div(
                tf.sub(x, mu),
                tf.mul(sigma, Math.sqrt(2.0))
            )
            return tf.mul(0.5, tf.add(1.0, tf.erf(adjustedInput)))
        })
    }
}

tf.serialization.registerClass(LaplaceActivation)

const activations = {
    APTx: new APTx(),
    Snake: new Snake(),
    Laplace: new LaplaceActivation()
}

export default activations

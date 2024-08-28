import * as tf from '@tensorflow/tfjs'

/**
 * Laplace activation function
 * https://arxiv.org/abs/2209.10655
 */
export default class Laplace extends tf.serialization.Serializable {
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

tf.serialization.registerClass(Laplace)

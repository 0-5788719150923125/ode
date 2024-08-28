import * as tf from '@tensorflow/tfjs'

/**
 * Snake activation function
 * https://arxiv.org/abs/2309.07803
 */
export default class Snake extends tf.serialization.Serializable {
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

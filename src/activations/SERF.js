import * as tf from '@tensorflow/tfjs'

/**
 * SERF activation function
 * https://arxiv.org/abs/2108.09598
 */
export default class SERF extends tf.serialization.Serializable {
    /** @nocollapse */
    static className = 'serf'

    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x) {
        return tf.tidy(() => {
            const expTerm = tf.exp(x)
            const logTerm = tf.log1p(expTerm)
            const erfTerm = tf.erf(logTerm)
            return tf.mul(x, erfTerm)
        })
    }
}

tf.serialization.registerClass(SERF)

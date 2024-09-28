import * as tf from '@tensorflow/tfjs'

export default function klDivergence(labels, predictions) {
    return tf.tidy(() => {
        const eps = 1e-8 // small value to avoid log(0)
        const safeLabels = tf.clipByValue(labels, eps, 1)
        const safePredictions = tf.clipByValue(predictions, eps, 1)
        return tf
            .sum(
                tf.mul(safeLabels, tf.log(tf.div(safeLabels, safePredictions)))
            )
            .mean()
    })
}

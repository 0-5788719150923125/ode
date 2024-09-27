import * as tf from '@tensorflow/tfjs'

export default function meanAbsoluteError(labels, predictions, weights = null) {
    return tf.tidy(() => {
        return tf.losses.absoluteDifference(
            labels,
            predictions,
            weights,
            tf.Reduction.MEAN
        )
    })
}

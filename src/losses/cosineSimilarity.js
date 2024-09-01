import * as tf from '@tensorflow/tfjs'

export default function cosineSimilarity(
    labels,
    predictions,
    axis = -1,
    weights = null,
    reduction = tf.Reduction.MEAN
) {
    return tf.tidy(() => {
        return tf.sub(
            1,
            tf.losses.cosineDistance(
                labels,
                predictions,
                axis,
                weights,
                reduction
            )
        )
    })
}

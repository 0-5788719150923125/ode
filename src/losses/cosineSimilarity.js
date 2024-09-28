import * as tf from '@tensorflow/tfjs'

export default function cosineSimilarity(
    labels,
    predictions,
    axis = -1,
    weights = null,
    reduction = tf.Reduction.MEAN
) {
    return tf.tidy(() => {
        const similarity = tf.sub(
            1,
            tf.losses.cosineDistance(
                labels,
                predictions,
                axis,
                weights,
                reduction
            )
        )
        return tf.add(similarity, 1) // Add 1 to shift the range from [-1, 1] to [0, 2]
    })
}

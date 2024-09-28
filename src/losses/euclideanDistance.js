import * as tf from '@tensorflow/tfjs'

export default function euclideanDistance(labels, predictions) {
    return tf.tidy(() => {
        const squaredDifferences = tf.square(tf.sub(labels, predictions))
        return tf.sqrt(tf.sum(squaredDifferences)).mean()
    })
}

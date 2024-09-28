import * as tf from '@tensorflow/tfjs'

export default function manhattanDistance(labels, predictions) {
    return tf.tidy(() => {
        return tf.sum(tf.abs(tf.sub(labels, predictions))).mean()
    })
}

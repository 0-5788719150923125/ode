import * as tf from '@tensorflow/tfjs'

export default function cosineSimilarity(t1, t2) {
    return tf.tidy(() => {
        // Determine the target shape
        const targetShape = [
            Math.max(t1.shape[0], 1),
            Math.max(t1.shape[1], t2.shape[0]),
            Math.max(t1.shape[2], t2.shape[1])
        ]

        // Pad tensors to match the target shape
        const padded1 = t1.pad(
            [
                [0, targetShape[0] - t1.shape[0]],
                [0, targetShape[1] - t1.shape[1]],
                [0, targetShape[2] - t1.shape[2]]
            ],
            0
        ) // Pad with zeros
        const padded2 = t2.expandDims(0).pad(
            [
                [0, targetShape[0] - 1],
                [0, targetShape[1] - t2.shape[0]],
                [0, targetShape[2] - t2.shape[1]]
            ],
            0
        ) // Pad with zeros

        // Compute dot product
        const dotProduct = tf.sum(tf.mul(padded1, padded2))

        // Compute norms
        const norm1 = tf.sqrt(tf.sum(tf.square(padded1)))
        const norm2 = tf.sqrt(tf.sum(tf.square(padded2)))

        // Compute cosine similarity
        return dotProduct.div(norm1.mul(norm2))
    })
}

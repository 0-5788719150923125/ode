import * as tf from '@tensorflow/tfjs'

export default function cosineSimilarity(t1, t2) {
    return tf.tidy(() => {
        // Flatten the tensors
        const flat1 = t1.reshape([-1])
        const flat2 = t2.reshape([-1])

        // Determine the length to use (minimum of the two flattened tensors)
        const minLength = Math.min(flat1.shape[0], flat2.shape[0])

        // Slice the tensors to the common length
        const slice1 = flat1.slice([0], [minLength])
        const slice2 = flat2.slice([0], [minLength])

        // Compute dot product
        const dotProduct = tf.sum(tf.mul(slice1, slice2))

        // Compute norms
        const norm1 = tf.sqrt(tf.sum(tf.square(slice1)))
        const norm2 = tf.sqrt(tf.sum(tf.square(slice2)))

        // Compute cosine similarity
        return tf.div(dotProduct, tf.maximum(tf.mul(norm1, norm2), 1e-8))
    })
}

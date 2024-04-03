import * as tf from '@tensorflow/tfjs'

function categoricalCrossentropy(target, output, fromLogits = false) {
    return tf.tidy(() => {
        if (fromLogits) {
            output = tf.softmax(output)
        } else {
            // scale preds so that the class probabilities of each sample sum to 1.
            const outputSum = tf.sum(output, output.shape.length - 1, true)
            output = tf.div(output, outputSum)
        }
        const epsilon = 1e-7
        output = tf.clipByValue(output, epsilon, 1 - epsilon)
        return tf.neg(
            tf.sum(
                tf.mul(tf.cast(target, 'float32'), tf.log(output)),
                output.shape.length - 1
            )
        )
    })
}

function sparseCategoricalCrossentropy(target, output, fromLogits = false) {
    return tf.tidy(() => {
        // Ensure the target is a flat array of integers
        const flatTarget = target.flatten().toInt()

        // Clip the output predictions to avoid log(0) error
        const epsilon = 1e-7
        output = output.clipByValue(epsilon, 1 - epsilon)

        // Determine the number of classes from the output shape
        const numClasses = output.shape[output.shape.length - 1]

        // One-hot encode the flatTarget
        const oneHotTarget = tf
            .oneHot(flatTarget, numClasses)
            .reshape(output.shape)

        // Calculate the categorical crossentropy
        let loss
        if (fromLogits) {
            loss = tf.losses.softmaxCrossEntropy(oneHotTarget, output)
        } else {
            loss = categoricalCrossentropy(oneHotTarget, output, fromLogits)
        }

        return loss.mean()
    })
}

function categoricalFocalCrossEntropy(
    yTrue,
    yPred,
    weights,
    labelSmoothing = 0,
    reduction = tf.Reduction.SUM_BY_NONZERO_WEIGHTS,
    alpha = null,
    gamma = 2.0,
    fromLogits = false,
    axis = -1
) {
    return tf.tidy(() => {
        // Clip values to prevent division by zero error
        const epsilon = tf.backend().epsilon()
        const clippedPred = tf.clipByValue(yPred, epsilon, 1 - epsilon)

        if (fromLogits) {
            yPred = tf.softmax(yPred, axis)
        }

        // Calculate cross-entropy loss
        const crossEntropy = tf.mul(tf.neg(yTrue), tf.log(clippedPred))

        let focalLoss
        if (alpha !== null) {
            const alphaWeight = tf.scalar(alpha)
            focalLoss = tf.mul(
                tf.mul(alphaWeight, tf.pow(tf.sub(1, clippedPred), gamma)),
                crossEntropy
            )
        } else {
            focalLoss = tf.mul(
                tf.pow(tf.sub(1, clippedPred), gamma),
                crossEntropy
            )
        }

        return tf.mean(tf.sum(focalLoss, -1))
    })
}

const customLosses = {
    categoricalCrossentropy: (target, output, fromLogits) =>
        categoricalCrossentropy(target, output, fromLogits),
    sparseCategoricalCrossentropy: (target, output, fromLogits) =>
        sparseCategoricalCrossentropy(target, output, fromLogits),
    categoricalFocalCrossEntropy: categoricalFocalCrossEntropy
}

export default customLosses

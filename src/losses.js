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
    target,
    output,
    weights,
    labelSmoothing = 0,
    reduction = tf.Reduction.SUM_BY_NONZERO_WEIGHTS,
    alpha = 0.25,
    gamma = 2.0,
    fromLogits = false,
    axis = -1
) {
    return tf.tidy(() => {
        if (fromLogits) {
            output = tf.softmax(output, axis)
        }

        const numClasses = target.shape[axis]
        const smoothedTarget = tf.add(
            tf.mul(tf.cast(target, 'float32'), 1 - labelSmoothing),
            labelSmoothing / numClasses
        )

        const alphaTensor = Array.isArray(alpha)
            ? tf.tensor(alpha, target.shape)
            : tf.scalar(alpha)

        const pt = tf.where(
            tf.equal(smoothedTarget, 1),
            output,
            tf.sub(1, output)
        )
        const modulator = tf.pow(tf.sub(1, pt), tf.scalar(gamma))
        const focalLoss = tf.neg(
            tf.mul(tf.mul(alphaTensor, modulator), tf.log(pt))
        )

        if (reduction === tf.Reduction.NONE) {
            return focalLoss
        } else if (reduction === tf.Reduction.SUM) {
            return tf.sum(focalLoss)
        } else if (reduction === tf.Reduction.MEAN) {
            return tf.mean(focalLoss)
        } else if (reduction === tf.Reduction.SUM_BY_NONZERO_WEIGHTS) {
            const numNonZeros = tf.sum(tf.notEqual(target, 0))
            return tf.div(tf.sum(focalLoss), numNonZeros)
        } else {
            throw new Error(`Unknown reduction: ${reduction}`)
        }
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

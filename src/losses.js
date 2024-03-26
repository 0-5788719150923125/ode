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

function categoricalFocalLoss(gamma = 2.0, alpha = 1.0) {
    return function (yTrue, yPred) {
        return tf.tidy(() => {
            // Ensure predictions are probabilities
            const epsilon = tf.backend().epsilon()
            yPred = tf.clipByValue(yPred, epsilon, 1 - epsilon)

            // Calculate the cross entropy component of the focal loss
            const crossEntropy = yTrue.mul(yPred.log().neg())

            // Calculate the focal loss's modulating factor
            const pT = yTrue
                .mul(yPred)
                .add(yTrue.neg().add(1).mul(yPred.neg().add(1)))
            const modulatingFactor = pT.pow(gamma)

            // Apply the alpha weighting factor
            const alphaTensor = yTrue.mul(alpha).add(
                yTrue
                    .neg()
                    .add(1)
                    .mul(1 - alpha)
            )
            const focalLoss = alphaTensor
                .mul(modulatingFactor)
                .mul(crossEntropy)

            // Reduce the loss to a single scalar value
            return focalLoss.mean()
        })
    }
}

function sparseCategoricalFocalLoss(gamma, fromLogits = false) {
    return (yTrue, yPred) => {
        return tf.tidy(() => {
            // Ensure yPred is probabilities (not logits) if fromLogits is false
            if (fromLogits) {
                yPred = tf.softmax(yPred)
            }

            const epsilon = tf.backend().epsilon()
            const yPredClipped = tf.clipByValue(yPred, epsilon, 1 - epsilon)

            // Calculate the cross entropy
            const crossEntropy = tf.neg(
                tf.sum(tf.mul(yTrue, tf.log(yPredClipped)), -1)
            )

            // Calculate p_t
            const pT = tf.add(
                tf.mul(yTrue, yPredClipped),
                tf.mul(tf.sub(1, yTrue), tf.sub(1, yPredClipped))
            )

            // Calculate the focal loss scaling factor
            const alphaFactor = tf.scalar(1) // Change as needed
            const modulatingFactor = tf.pow(tf.sub(1.0, pT), gamma)

            // Calculate focal loss
            const focalLoss = tf.mul(
                alphaFactor,
                tf.mul(modulatingFactor, crossEntropy)
            )

            // Reduce over all elements to get final loss
            const loss = tf.mean(focalLoss)
            return loss
        })
    }
}

const customLosses = {
    categoricalCrossentropy: (target, output, fromLogits) =>
        categoricalCrossentropy(target, output, fromLogits),
    sparseCategoricalCrossentropy: (target, output, fromLogits) =>
        sparseCategoricalCrossentropy(target, output, fromLogits),
    categoricalFocalLoss: (gamma, alpha) => categoricalFocalLoss(gamma, alpha),
    sparseCategoricalFocalLoss: (gamma, fromLogits) =>
        sparseCategoricalFocalLoss(gamma, fromLogits)
}

export default customLosses

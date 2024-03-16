import * as tf from '@tensorflow/tfjs'

export function categoricalCrossentropy(target, output, fromLogits = false) {
    return tf.tidy(() => {
        if (fromLogits) {
            output = tf.softmax(output)
        } else {
            // scale preds so that the class probabilities of each sample sum to 1.
            const outputSum = tf.sum(output, output.shape.length - 1, true)
            output = tf.div(output, outputSum)
        }
        output = tf.clipByValue(output, 1e-7, 1 - 1e-7)
        return tf.neg(
            tf.sum(
                tf.mul(tf.cast(target, 'float32'), tf.log(output)),
                output.shape.length - 1
            )
        )
    })
}

export function sparseCategoricalCrossentropy(
    target,
    output,
    fromLogits = false
) {
    return tf.tidy(() => {
        const flatTarget = tf.cast(tf.floor(K.flatten(target)), 'int32')
        output = tf.clipByValue(output, 1e-7, 1 - 1e-7)
        const outputShape = output.shape
        const oneHotTarget = tf.reshape(
            tf.oneHot(flatTarget, outputShape[outputShape.length - 1]),
            outputShape
        )
        return categoricalCrossentropy(oneHotTarget, output, fromLogits)
    })
}

export function focalLoss(gamma = 2.0, alpha = 0.25) {
    return function (yTrue, yPred) {
        // Ensure predictions are probabilities (not logits)
        yPred = yPred.softmax()

        // Clip the predictions to prevent log(0)
        const epsilon = 1e-7
        yPred = yPred.clipByValue(epsilon, 1 - epsilon)

        // Calculate Cross Entropy
        const crossEntropy = yTrue.mul(yPred.log()).mul(-1)

        // Calculate modulating factor and apply alpha weighting
        const pT = yTrue
            .mul(yPred)
            .add(yTrue.mul(-1).add(1).mul(yPred.mul(-1).add(1)))
        const modulatingFactor = pT.mul(-1).add(1).pow(gamma)
        const alphaFactor = yTrue.mul(alpha).add(
            yTrue
                .mul(-1)
                .add(1)
                .mul(1 - alpha)
        )
        const focalLoss = alphaFactor.mul(modulatingFactor).mul(crossEntropy)

        // Reduce the loss to a single scalar value
        return focalLoss.mean()
    }
}

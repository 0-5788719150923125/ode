import * as tf from '@tensorflow/tfjs'

// Focal loss is used to address the issue of the class imbalance problem.
// A modulation term applied to the Cross-Entropy loss function.
// https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
export default function categoricalFocalCrossEntropy(
    yTrue,
    yPred,
    weights = null,
    labelSmoothing = 0,
    reduction = null,
    fromLogits = false,
    alpha = 1.0, // alpha > 0.5 penalises false negatives more than false positives
    gamma = 2.0 // focal parameter controls degree of down-weighting of easy examples
) {
    return tf.tidy(() => {
        // Clip the prediction value to prevent NaN's and Inf's
        const epsilon = tf.backend().epsilon()
        yPred = tf.clipByValue(yPred, epsilon, 1.0 - epsilon)

        // Calculate probabilities
        if (fromLogits) yPred = tf.softmax(yPred)

        // Calculate cross entropy manually
        let crossEntropy = tf.mul(tf.neg(yTrue), tf.log(yPred))

        // Apply label smoothing if needed
        if (labelSmoothing > 0) {
            const numClasses = yTrue.shape[1]
            yTrue = tf.add(
                tf.mul(yTrue, tf.sub(1.0, labelSmoothing)),
                tf.div(labelSmoothing, numClasses)
            )
        }

        // Calculate focal loss
        const modulation = tf.pow(tf.sub(1.0, yPred), gamma)
        let losses = tf.mul(alpha, tf.mul(modulation, crossEntropy))

        if (weights !== null) {
            losses = tf.mul(losses, weights.expandDims(-1))
        }

        // Compute scalar loss with reduction
        return tf.mean(tf.sum(losses, -1))
    })
}

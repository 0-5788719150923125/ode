import * as tf from '@tensorflow/tfjs'

// This loss function not only addresses the issue of noisy labels in training,
// but also enhances model confidence calibration and helps with training regularization.
// https://medium.com/ntropy-network/is-cross-entropy-all-you-need-lets-discuss-an-alternative-ac0df6ff5691
export default function smoothGeneralizedCrossEntropy(
    yTrue,
    yPred,
    weights = null,
    labelSmoothing = 0,
    reduction = tf.Reduction.MEAN,
    fromLogits = false,
    alpha = null,
    gamma = null,
    sigma = null,
    epsilon = 1e-8,
    q = 0.5
) {
    return tf.tidy(() => {
        // Ensure yPred is probabilities
        const pred = fromLogits ? tf.softmax(yPred) : yPred

        // Clip probabilities to avoid numerical instability
        const predClipped = tf.clipByValue(pred, epsilon, 1.0 - epsilon)

        // Calculate the numerator part of the GCE loss
        const numerator = tf.pow(predClipped, q)

        // Make the numerator more numerically stable
        const predStable = tf.where(
            tf.greaterEqual(predClipped, 0),
            tf.add(numerator, epsilon),
            tf.neg(tf.add(tf.pow(tf.abs(predClipped), q), epsilon))
        )

        // Calculate the denominator part of the GCE loss
        const loss = tf.div(tf.sub(1, predStable), q + epsilon)

        // Apply label smoothing
        const numClasses = pred.shape[pred.shape.length - 1]
        const smoothingValue = labelSmoothing / (numClasses - 1)
        const smoothedLabels = tf.add(
            tf.mul(yTrue, 1 - labelSmoothing),
            tf.mul(tf.onesLike(yTrue), smoothingValue)
        )

        // Apply smoothing to the loss
        const smoothedLoss = tf.mul(smoothedLabels, loss)

        return tf.losses.computeWeightedLoss(smoothedLoss, weights, reduction)

        // // Apply sample weights if provided
        // let weightedLoss = smoothedLoss
        // if (weights !== null) {
        //     weightedLoss = tf.mul(weightedLoss, weights.expandDims(-1))
        // }

        // // Mean reduction to collapse the loss into a single number
        // return tf.mean(tf.sum(weightedLoss, -1))
    })
}

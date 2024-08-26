import * as tf from '@tensorflow/tfjs'

// Focal loss is used to address the issue of the class imbalance problem.
// A modulation term applied to the Cross-Entropy loss function.
// https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
function categoricalFocalCrossEntropy(
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

// This loss function not only addresses the issue of noisy labels in training,
// but also enhances model confidence calibration and helps with training regularization.
// https://medium.com/ntropy-network/is-cross-entropy-all-you-need-lets-discuss-an-alternative-ac0df6ff5691
function smoothGeneralizedCrossEntropy(
    yTrue,
    yPred,
    weights = null,
    labelSmoothing = 0,
    reduction = null,
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

        // Apply sample weights if provided
        let weightedLoss = smoothedLoss
        if (weights !== null) {
            weightedLoss = tf.mul(weightedLoss, weights.expandDims(-1))
        }

        // Mean reduction to collapse the loss into a single number
        return tf.mean(tf.sum(weightedLoss, -1))
    })
}

// mitigating the bias of learning difficulties with tokens
// https://github.com/suu990901/LLaMA-MiLe-Loss/blob/main/utils/trainer.py
function MiLeCrossEntropy(
    yTrue,
    yPred,
    weights = null,
    labelSmoothing = 0,
    reduction = tf.Reduction.MEAN,
    fromLogits = false,
    alpha = null,
    gamma = 1.0,
    sigma = 1.0,
    epsilon = 1e-8
) {
    return tf.tidy(() => {
        // Ensure yPred is logits
        const logits = fromLogits ? yPred : tf.logSoftmax(yPred)

        // Calculate cross-entropy loss
        let ceLoss = tf.losses.softmaxCrossEntropy(
            yTrue,
            logits,
            undefined,
            labelSmoothing,
            tf.Reduction.NONE,
            fromLogits
        )

        // Calculate entropy
        const probs = fromLogits ? tf.softmax(logits) : yPred
        const clippedProbs = tf.clipByValue(probs, epsilon, 1.0)
        const entropy = tf.sum(
            tf.neg(tf.mul(clippedProbs, tf.log(clippedProbs)))
        )

        // Calculate alpha (normalization factor)
        const aDenom = tf.mean(tf.pow(tf.add(sigma, entropy), gamma))
        const alpha = tf.div(1.0, aDenom)

        // Calculate final loss
        let losses = tf.mul(
            alpha,
            tf.mul(tf.pow(tf.add(sigma, entropy), gamma), ceLoss)
        )

        // Apply reduction
        return tf.losses.computeWeightedLoss(losses, weights, reduction)
    })
}

const customLosses = {
    categoricalFocalCrossEntropy,
    MiLeCrossEntropy,
    smoothGeneralizedCrossEntropy,
    softmaxCrossEntropy: tf.losses.softmaxCrossEntropy
}

export default customLosses

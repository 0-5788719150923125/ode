import * as tf from '@tensorflow/tfjs'

// mitigating the bias of learning difficulties with tokens
// https://arxiv.org/abs/2310.19531
// https://github.com/suu990901/LLaMA-MiLe-Loss/blob/main/utils/trainer.py
export default function MiLeCrossEntropy(
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
        let crossEntropy = tf.losses.softmaxCrossEntropy(
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
            tf.mul(tf.pow(tf.add(sigma, entropy), gamma), crossEntropy)
        )

        // Apply reduction
        return tf.losses.computeWeightedLoss(losses, weights, reduction)
    })
}

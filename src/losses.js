import * as tf from '@tensorflow/tfjs'

// https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
// function categoricalFocalCrossEntropy(
//     y_true,
//     y_pred,
//     weights = undefined,
//     labelSmoothing = 0,
//     reduction = tf.Reduction.MEAN,
//     alpha = 0.25,
//     gamma = 2.0,
//     fromLogits = false
// ) {
//     return tf.tidy(() => {
//         // Apply label smoothing if needed
//         if (labelSmoothing > 0) {
//             const numClasses = tf.scalar(y_true.shape[1])
//             y_true = tf.add(
//                 tf.mul(y_true, tf.sub(1.0, labelSmoothing)),
//                 tf.div(labelSmoothing, numClasses)
//             )
//         }

//         // Calculate softmax cross-entropy loss
//         const crossEntropy = tf.losses.softmaxCrossEntropy(
//             y_true,
//             y_pred,
//             undefined,
//             undefined,
//             tf.Reduction.NONE,
//             fromLogits
//         )

//         // Calculate focal loss
//         const alpha_t = tf
//             .mul(y_true, alpha)
//             .add(tf.mul(tf.sub(1.0, y_true), 1.0 - alpha))
//         const modulation_factor = tf.pow(tf.sub(1.0, tf.softmax(y_pred)), gamma)
//         const focalLoss = tf.mul(
//             alpha_t,
//             tf.mul(modulation_factor, crossEntropy.expandDims(-1))
//         )

//         // Compute weighted loss with reduction
//         return tf.losses.computeWeightedLoss(focalLoss, weights, reduction)
//     })
// }

function categoricalFocalCrossEntropy(
    y_true,
    y_pred,
    weights = null,
    labelSmoothing = 0,
    reduction = null,
    fromLogits = false,
    alpha = 0.25,
    gamma = 2.0
) {
    return tf.tidy(() => {
        // Clip the prediction value to prevent NaN's and Inf's
        const epsilon = tf.backend().epsilon()
        y_pred = tf.clipByValue(y_pred, epsilon, 1.0 - epsilon)

        // Calculate probabilities
        if (fromLogits) y_pred = tf.softmax(y_pred)

        // Calculate cross entropy manually
        let cross_entropy = tf.mul(tf.neg(y_true), tf.log(y_pred))

        // Apply label smoothing if needed
        if (labelSmoothing > 0) {
            const numClasses = y_true.shape[1]
            y_true = tf.add(
                tf.mul(y_true, tf.sub(1.0, labelSmoothing)),
                tf.div(labelSmoothing, numClasses)
            )
        }

        // Calculate focal loss
        const alpha_t = tf
            .mul(y_true, alpha)
            .add(tf.mul(tf.sub(1.0, y_true), 1.0 - alpha))
        const modulation_factor = tf.pow(tf.sub(1.0, y_pred), gamma)
        let focal_loss = tf.mul(
            alpha_t,
            tf.mul(modulation_factor, cross_entropy)
        )

        if (weights !== null) {
            focal_loss = tf.mul(focal_loss, weights.expandDims(-1))
        }

        // Compute scalar loss with reduction
        return tf.mean(tf.sum(focal_loss, -1))
    })
}

const customLosses = {
    softmaxCrossEntropy: tf.losses.softmaxCrossEntropy,
    categoricalFocalCrossEntropy
}

export default customLosses

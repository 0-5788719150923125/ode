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
        const epsilon = tf.backend().epsilon()
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
        const epsilon = tf.backend().epsilon()
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

// https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
function categoricalFocalCrossEntropy(
    target,
    output,
    weights,
    labelSmoothing = 0,
    reduction = tf.Reduction.MEAN,
    alpha = null,
    gamma = 2.0,
    fromLogits = false,
    axis = -1
) {
    return tf.tidy(() => {
        // Clip values to prevent division by zero error
        const epsilon = tf.backend().epsilon()
        const clippedPred = tf.clipByValue(output, epsilon, 1 - epsilon)

        if (fromLogits) {
            output = tf.softmax(output, axis)
        }

        // Calculate cross-entropy loss
        const crossEntropy = tf.losses.softmaxCrossEntropy(
            target,
            clippedPred,
            null,
            labelSmoothing
        )

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

        // return tf.mean(tf.sum(focalLoss, -1))
        return tf.losses.computeWeightedLoss(focalLoss, weights, reduction)
    })
}

// function softmaxCrossEntropy_<T extends Tensor, O extends Tensor>(
//     onehotLabels: T|TensorLike, logits: T|TensorLike,
//     weights?: Tensor|TensorLike, labelSmoothing = 0,
//     reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
//   let $onehotLabels =
//       convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
//   const $logits = convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
//   let $weights: Tensor = null;

//   if (weights != null) {
//     $weights = convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
//   }

//   assertShapesMatch(
//       $onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');

//   if (labelSmoothing > 0) {
//     const labelSmoothingScalar = scalar(labelSmoothing);
//     const one = scalar(1);
//     const numClasses = scalar($onehotLabels.shape[1]);

//     $onehotLabels =
//         add(mul($onehotLabels, sub(one, labelSmoothingScalar)),
//             div(labelSmoothingScalar, numClasses));
//   }

//   const losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);

//   return computeWeightedLoss(losses, $weights, reduction);
// }

const customLosses = {
    softmaxCrossEntropy: tf.losses.softmaxCrossEntropy,
    categoricalCrossentropy,
    sparseCategoricalCrossentropy,
    categoricalFocalCrossEntropy
}

export default customLosses

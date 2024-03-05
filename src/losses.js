import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

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

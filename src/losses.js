import * as tfjs from '@tensorflow/tfjs'

let tf = tfjs

;(async function () {
    if (typeof window === 'undefined') {
        tf = await import('@tensorflow/tfjs-node-gpu')
    }
})()

export function focalLoss(gamma = 2.0, alpha = 0.25) {
    return (yTrue, yPred) => {
        const epsilon = 1e-7
        yPred = yPred.clipByValue(epsilon, 1 - epsilon)
        const alphaTensor = tf.fill(yPred.shape, alpha)
        const modulatingFactor = tf.pow(tf.sub(1, yPred), gamma)
        const alphaModulatedLoss = tf.mul(
            alphaTensor,
            tf.mul(modulatingFactor, tf.mul(yTrue, tf.log(yPred)))
        )
        const focalLoss = tf.mul(-1, alphaModulatedLoss)
        return focalLoss.mean()
    }
}

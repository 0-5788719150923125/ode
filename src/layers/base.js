import * as tf from '@tensorflow/tfjs'
import customOps from '../ops.js'

export default class LayerBase extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.ode = {
            ops: customOps
        }
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            // stuff should go here
        })
    }

    // Direct application of matMul to x and kernel throws:
    // > Error in gradient for op BatchMatMul.
    // > The gradient of input 'b' has shape '16,48,48',
    // > which does not match the shape of the input '48,48'
    // Two solutions worked:
    // 1. Use tf.layers.dense but reassign kernel and bias
    // 2. Use tf.matMul but expandDims and tile kernel (current)
    // Another option, of course, is to separate attention logic
    // from trainable weights completely and use tf.layers.dense
    // inside a model definition. I was not able to define fully
    // function regular dense layers inside a custom layer.
    // Something related to how weights are loaded with this.kernel
    // and duplicating names

    applyDense(x, kernel, bias) {
        let k = kernel.expandDims(0).tile([x.shape[0], 1, 1])
        const m = tf.matMul(x, k)
        if (bias) return tf.add(m, bias)
        else return m
    }

    // rmsNorm = (x) => {
    //     const rms = tf.sqrt(tf.mean(tf.square(x), -1, true))
    //     const epsilon = 1e-7
    //     return x.div(rms.add(epsilon))
    // }

    // findLayer(key) {
    //     const lowercaseKey = key.toLowerCase()
    //     const match = Object.keys(customLayers).find(
    //         (k) => k.toLowerCase() === lowercaseKey
    //     )
    //     return match ? customLayers[match] : undefined
    // }

    applyALiBi(scores, numHeads, currentHead, seqLen) {
        if (!this.alibiSlope) {
            const base = tf.scalar(2 ** 8)
            const powers = tf
                .range(0, numHeads)
                .cast('float32')
                .add(tf.scalar(1))
            const slopes = tf.pow(base, powers.div(tf.scalar(numHeads)))
            this.alibiSlope = tf.keep(
                slopes.reciprocal().expandDims(-1).expandDims(-1)
            )
        }
        const alibiSlope = this.alibiSlope.gather([currentHead])
        const range = tf.range(0, seqLen)
        const relativePositions = range.expandDims(1).sub(range.expandDims(0))
        const alibiScores = tf.mul(alibiSlope, relativePositions)

        const adjustedAlibiScores = alibiScores.slice(
            [0, 0, 0],
            [1, seqLen, scores.shape[2]]
        )
        const expandedAlibiScores = adjustedAlibiScores.tile([
            scores.shape[0],
            1,
            1
        ])

        return scores.add(expandedAlibiScores)
    }

    static get className() {
        return this.name
    }

    getConfig() {
        return {
            ...super.getConfig(),
            className: this.getClassName()
        }
    }
}

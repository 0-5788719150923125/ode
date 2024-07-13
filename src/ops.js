import * as tf from '@tensorflow/tfjs'

// Helper function to generate Gumbel noise
function sampleGumbel(shape, epsilon = 1e-8) {
    const uniform = tf.randomUniform(shape, 0, 1)
    return tf.neg(tf.log(tf.neg(tf.log(uniform.add(epsilon)))))
}

// Helper function to apply Gumbel-Softmax trick
function gumbelSoftmax(logits, temperature = 1.0) {
    const gumbelNoise = sampleGumbel(logits.shape)
    return tf.softmax(logits.add(gumbelNoise).div(temperature))
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

function applyDense(x, kernel, bias) {
    let k = kernel.expandDims(0).tile([x.shape[0], 1, 1])
    const m = tf.matMul(x, k)
    if (bias) return tf.add(m, bias)
    else return m
}

function rmsNorm(x) {
    const rms = tf.sqrt(tf.mean(tf.square(x), -1, true))
    const epsilon = 1e-7
    return x.div(rms.add(epsilon))
}

function applyALiBi(scores, numHeads, currentHead, seqLen, maxSeqLen = 2048) {
    if (!this.alibiSlopes) {
        const slopesPerHead = tf.pow(
            tf.scalar(2),
            tf.range(0, numHeads).cast('float32').div(tf.scalar(numHeads))
        )
        const slopesPerPos = tf
            .range(0, maxSeqLen)
            .cast('float32')
            .expandDims(0)
        this.alibiSlopes = tf.keep(
            slopesPerHead.expandDims(1).mul(slopesPerPos)
        )
    }

    const alibiScores = this.alibiSlopes
        .slice([currentHead, 0], [1, seqLen])
        .expandDims(0)

    return scores.sub(alibiScores)
}

export default {
    gumbelSoftmax,
    rmsNorm,
    applyALiBi,
    applyDense
}

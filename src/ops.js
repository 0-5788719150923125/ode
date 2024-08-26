import * as tf from '@tensorflow/tfjs'
import { LinearCongruentialGenerator } from './utils.js'

// Singleton implementation
const seedGenerator = (function () {
    let instance
    let minValue
    let maxValue

    function createInstance(seed, min, max) {
        minValue = min
        maxValue = max
        return new LinearCongruentialGenerator(seed)
    }

    return {
        setSeed: function (seed, min, max) {
            if (seed === undefined || min === undefined || max === undefined) {
                throw new Error(
                    'seed, min, and max must be provided to initialize'
                )
            }
            instance = createInstance(seed, min, max)
        },
        getSeed: function (min, max) {
            if (!instance) {
                console.warn(
                    'Generator not seeded. Initializing weights with random values.'
                )
                return undefined
            }
            if (min !== undefined && max !== undefined) {
                minValue = min
                maxValue = max
            }
            return instance.randomFloat(minValue, maxValue)
        }
    }
})()

export const { setSeed, getSeed } = seedGenerator

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
// 2. Use tf.matMul but expandDims and tile kernel (this)
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

function applyALiBi(scores, numHeads, queriesPerHead, maxSeqLen = 2048) {
    return tf.tidy(() => {
        const [batchSize, _, __, queryLen, keyLen] = scores.shape

        // Generate slopes for each head
        const slopesPerHead = tf.pow(
            tf.scalar(2),
            tf.range(0, numHeads).add(1).mul(-8).div(tf.scalar(numHeads))
        )

        // Generate position indices
        const positions = tf.range(0, maxSeqLen).cast('float32')

        // Compute ALiBi slopes for all heads and positions
        const alibiSlopes = tf.outerProduct(slopesPerHead, positions)

        // Slice the slopes to match the current sequence length
        const slicedSlopes = alibiSlopes.slice([0, 0], [numHeads, keyLen])

        // Reshape to match scores dimensions and broadcast
        const alibiScores = slicedSlopes
            .expandDims(1) // Add dimension for queries per head
            .expandDims(1) // Add dimension for queries
            .expandDims(0) // Add dimension for batch
            .broadcastTo([
                batchSize,
                numHeads,
                queriesPerHead,
                queryLen,
                keyLen
            ])

        // Subtract ALiBi scores from attention scores
        return scores.sub(alibiScores)
    })
}

function zip(tensor1, tensor2) {
    // Ensure tensors have the same shape
    if (!tf.util.arraysEqual(tensor1.shape, tensor2.shape)) {
        throw new Error('Tensors must have the same shape')
    }

    // Reshape tensors to 2D: [numElements, 1]
    const flatShape = [-1, 1]
    const flat1 = tensor1.reshape(flatShape)
    const flat2 = tensor2.reshape(flatShape)

    // Stack the flattened tensors
    const stacked = tf.stack([flat1, flat2])

    // Reshape to interleave: [numElements * 2, 1]
    const interleaved = stacked.reshape([-1])

    // Reshape back to original shape, but with twice the size in last dimension
    const newShape = tensor1.shape.slice()
    newShape[newShape.length - 1] *= 2

    return interleaved.reshape(newShape)
}

export default {
    getSeed,
    setSeed,
    gumbelSoftmax,
    rmsNorm,
    applyALiBi,
    applyDense,
    zip
}

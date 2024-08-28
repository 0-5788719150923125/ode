import * as tf from '@tensorflow/tfjs'
import categoricalFocalCrossEntropy from './losses/categoricalFocalCrossEntropy.js'
import cosineSimilarity from './losses/cosineSimilarity.js'
import MiLeCrossEntropy from './losses/MiLeCrossEntropy.js'
import smoothGeneralizedCrossEntropy from './losses/smoothGeneralizedCrossEntropy.js'

// Create wrapped versions of all TFJS losses
const wrappedLosses = Object.fromEntries(
    Object.entries(tf.losses).map(([key, lossFunction]) => [key, lossFunction])
)

export default {
    ...wrappedLosses,
    categoricalFocalCrossEntropy,
    cosineSimilarity,
    MiLeCrossEntropy,
    smoothGeneralizedCrossEntropy
}

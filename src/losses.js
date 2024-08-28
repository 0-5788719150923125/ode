import * as tf from '@tensorflow/tfjs'
import categoricalFocalCrossEntropy from './losses/categoricalFocalCrossEntropy.js'
import cosineSimilarity from './losses/cosineSimilarity.js'
import MiLeCrossEntropy from './losses/MiLeCrossEntropy.js'
import smoothGeneralizedCrossEntropy from './losses/smoothGeneralizedCrossEntropy.js'

export default {
    categoricalFocalCrossEntropy,
    cosineSimilarity,
    MiLeCrossEntropy,
    smoothGeneralizedCrossEntropy,
    softmaxCrossEntropy: tf.losses.softmaxCrossEntropy
}

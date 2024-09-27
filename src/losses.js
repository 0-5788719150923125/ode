import * as tf from '@tensorflow/tfjs'
import categoricalFocalCrossEntropy from './losses/categoricalFocalCrossEntropy.js'
import cosineSimilarity from './losses/cosineSimilarity.js'
import meanAbsoluteError from './losses/meanAbsoluteError.js'
import MiLeCrossEntropy from './losses/MiLeCrossEntropy.js'
import smoothGeneralizedCrossEntropy from './losses/smoothGeneralizedCrossEntropy.js'

export default {
    ...tf.losses,
    categoricalFocalCrossEntropy,
    cosineSimilarity,
    meanAbsoluteError,
    MiLeCrossEntropy,
    smoothGeneralizedCrossEntropy
}

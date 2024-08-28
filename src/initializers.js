import * as tf from '@tensorflow/tfjs'
import { getSeed } from './ops.js'

const createInitializerFactory =
    (initializerConstructor) =>
    (config = {}) => {
        return initializerConstructor({
            ...config,
            seed: getSeed()
        })
    }

// Create wrapped versions of all TFJS initializers
const wrappedInitializers = Object.fromEntries(
    Object.entries(tf.initializers).map(([key, initializer]) => [
        key,
        createInitializerFactory(initializer)
    ])
)

export default wrappedInitializers

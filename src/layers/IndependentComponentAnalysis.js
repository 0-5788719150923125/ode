import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class IndependentComponentAnalysis extends LayerBase {
    constructor(config) {
        super(config)
        this.outputDim = config.outputDim
        this.maxIterations = config.maxIterations || 10
        this.tolerance = config.tolerance || 1e-6
        this.learningRate = config.learningRate || 0.01
    }

    build(inputShape) {
        this.inputDim = inputShape[inputShape.length - 1]
        this.kernelShape = [this.outputDim, this.inputDim]
        this.kernel = this.addWeight(
            'kernel',
            this.kernelShape,
            'float32',
            this.initializers.glorotNormal()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = inputs[0]
            const [batchSize, seqLength, featureDim] = input.shape

            // Reshape to 2D for processing
            const reshapedInput = input.reshape([-1, featureDim])

            // Normalize the input data
            const { mean, variance } = tf.moments(reshapedInput, 0)
            const normalizedInput = tf.div(
                tf.sub(reshapedInput, mean),
                tf.sqrt(variance)
            )

            const ica = this.fastICA(normalizedInput)
            const output = tf.matMul(normalizedInput, ica.transpose())

            // Reshape back to 3D
            return output.reshape([batchSize, seqLength, this.outputDim])
        })
    }

    fastICA(X) {
        let W = tf.randomNormal(
            [this.outputDim, X.shape[1]],
            0,
            1,
            'float32',
            42
        )

        for (let i = 0; i < this.maxIterations; i++) {
            const Wprev = W

            W = tf.tidy(() => {
                const WX = tf.matMul(W, X.transpose())
                const G = tf.tanh(WX)
                const Gder = tf.sub(1, tf.square(G))
                const GX = tf.matMul(G, X)
                const newW = tf.sub(
                    GX.div(X.shape[0]),
                    tf.mean(Gder, 1, true).mul(W)
                )

                // Apply the learning rate to the weight update
                const updateW = tf.mul(newW.sub(W), this.learningRate).add(W)

                // Normalize rows of W
                const rowNorms = tf.sqrt(tf.sum(tf.square(updateW), 1, true))
                return tf.div(updateW, rowNorms)
            })

            const distanceW = tf.mean(
                tf.abs(tf.sub(W, Wprev.slice([0, 0], W.shape)))
            )
            if (distanceW.bufferSync().get(0) < this.tolerance) {
                break
            }
        }

        return W
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            outputDim: this.outputDim,
            maxIterations: this.maxIterations,
            tolerance: this.tolerance,
            learningRate: this.learningRate
        }
    }
}

tf.serialization.registerClass(IndependentComponentAnalysis)

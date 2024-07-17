import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class AdaptiveMixtureOfExperts extends LayerBase {
    constructor(config) {
        super(config)
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.topK = config.topK || 2
        this.switchingDim = config?.switchingDim || 64
        this.activation = config.activation || 'swish'
        this.temperature = config.temperature || 1.0
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.switchingHidden = this.addWeight(
            'switchingHidden',
            [inputDim, this.switchingDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingHiddenBias = this.addWeight(
            'switchingHiddenBias',
            [this.switchingDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingKernel = this.addWeight(
            'switchingKernel',
            [this.switchingDim, this.numExperts],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.switchingBias = this.addWeight(
            'switchingBias',
            [this.numExperts],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.outputProjection = this.addWeight(
            'outputProjection',
            [this.topK * inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            return this.sparseTopKWithSTE(inputs, this.topK)
        })
    }

    sparseTopKWithSTE(inputs, k) {
        const switchingHidden = this.ops.applyDense(
            inputs,
            this.switchingHidden.read(),
            this.switchingHiddenBias.read()
        )
        const switchingActivated = tf.layers
            .activation({ activation: this.activation })
            .apply(switchingHidden)
        const switchingScores = this.ops.applyDense(
            switchingActivated,
            this.switchingKernel.read(),
            this.switchingBias.read()
        )

        const switchingScoresSummed = switchingScores.sum(1)
        const expertWeights = this.ops.gumbelSoftmax(
            switchingScoresSummed,
            this.temperature
        )

        let expertIndices
        const expertValues = tf.customGrad((expertWeights, save) => {
            const topK = tf.topk(expertWeights, k)
            expertIndices = tf.keep(topK.indices)
            save([expertWeights])
            return {
                value: topK.values,
                gradFunc: (dy, saved) => {
                    const [expertWeights] = saved
                    const tileShape = [1, expertWeights.shape[1] / k]
                    const tiledGradients = dy.tile(tileShape)
                    return [tiledGradients]
                }
            }
        })(expertWeights)

        const batchOutputs = []
        for (let i = 0; i < inputs.shape[0]; i++) {
            const batchInputs = inputs.slice([i, 0], [1, -1])
            const batchExpertIndices = expertIndices
                .slice([i, 0], [1, -1])
                .arraySync()[0]
            const batchExpertValues = expertValues.slice([i, 0], [1, -1])
            const expertOutputs = []

            for (let j = 0; j < k; j++) {
                const expertIndex = batchExpertIndices[j]
                const expertValue = batchExpertValues
                    .slice([0, j], [1, 1])
                    .squeeze()
                const expertOutput =
                    this.experts[expertIndex].apply(batchInputs)
                expertOutputs.push(expertOutput.mul(expertValue))
            }

            const concatenatedOutput = tf.concat(expertOutputs, -1)
            batchOutputs.push(concatenatedOutput)
        }

        const combinedOutput = tf.concat(batchOutputs, 0)

        const outputProjected = this.ops.applyDense(
            combinedOutput,
            this.outputProjection.read()
        )

        expertIndices.dispose()
        return outputProjected
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK,
            temperature: this.temperature
        }
    }
}

tf.serialization.registerClass(AdaptiveMixtureOfExperts)

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

        const outputs = tf.customGrad((inputs, switchingScores, save) => {
            const topK = tf.topk(switchingScores.sum(1), k)
            const indices = topK.indices.arraySync()

            const batchOutputs = []
            for (let i = 0; i < inputs.shape[0]; i++) {
                const batchInputs = inputs.slice([i, 0], [1, -1])
                const expertOutputs = []
                for (let j = 0; j < k; j++) {
                    const expertIndex = indices[i][j]
                    const expertOutput =
                        this.experts[expertIndex].apply(batchInputs)
                    expertOutputs.push(expertOutput)
                }
                const concatenatedOutput = tf.concat(expertOutputs, -1)
                batchOutputs.push(concatenatedOutput)
            }

            const combinedOutput = tf.concat(batchOutputs, 0)

            const outputProjected = this.ops.applyDense(
                combinedOutput,
                this.outputProjection.read()
            )

            save([inputs, switchingScores])

            return {
                value: outputProjected,
                gradFunc: (dy, saved) => {
                    const [origInputs, origSwitchingScores] = saved
                    return [dy, origSwitchingScores.mul(dy.mean())]
                }
            }
        })(inputs, switchingScores)

        return outputs.mul(switchingScores.mean())
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            switchingDim: this.switchingDim,
            activation: this.activation,
            topK: this.topK
        }
    }
}

tf.serialization.registerClass(AdaptiveMixtureOfExperts)

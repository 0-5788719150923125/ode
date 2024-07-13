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

        // Initialize switching network
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

            // Switching network
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

            // Select top-k experts for each batch
            const [batchSize, timeSteps, numExperts] = switchingScores.shape
            const linearWeights = tf
                .linspace(1, 2, timeSteps)
                .expandDims(0)
                .expandDims(-1)
            const weightedAvgScores = switchingScores
                .mul(linearWeights)
                .sum(1)
                .div(linearWeights.sum())

            const expertIndices = this.selectTopExperts(weightedAvgScores)

            // Predict on top-k experts, for every batch
            const batchOutputs = []
            for (let i = 0; i < inputs.shape[0]; i++) {
                const batchInputs = inputs.slice([i, 0], [1, -1])
                const expertOutputs = []
                for (let j = 0; j < this.topK; j++) {
                    const expertIndex = expertIndices[i][j]
                    const expertOutput =
                        this.experts[expertIndex].apply(batchInputs)
                    expertOutputs.push(expertOutput)
                }
                const concatenatedOutput = tf.concat(expertOutputs, -1)
                batchOutputs.push(concatenatedOutput)
            }

            // Concat expert outputs, and project them into the proper dimension
            const outputProjected = this.ops.applyDense(
                tf.concat(batchOutputs, 0),
                this.outputProjection.read()
            )

            return outputProjected
        })
    }

    selectTopExperts(switchingScores) {
        const topKIndices = tf.topk(switchingScores, this.topK).indices
        return topKIndices.arraySync()
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

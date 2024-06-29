import ODE from './model.v2.js'

/**
 * A better sparse mixture of experts.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 128
        this.numExperts = config.numExperts || 7
        this.topK = config.topK || 2
        this.switchingDim = config.switchingDim || 64
        this.headDim = config.headDim || 512
        this.mlpDim = config.mlpDim || 1024
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: 'OriginalDesign/thrice'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .AdaptiveMixtureOfExperts({
                    topK: this.topK,
                    switchingDim: this.switchingDim,
                    activation: 'mish',
                    experts: this.createAttentionExperts(outputs.shape)
                })
                .apply(outputs)

            outputs = this.ode.layers
                .AdaptiveMixtureOfExperts({
                    topK: this.topK,
                    numExperts: this.experts,
                    switchingDim: this.switchingDim,
                    activation: 'mish',
                    experts: this.createFeedforwardExperts(outputs.shape)
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    createAttentionExperts(inputShape) {
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                const expert = this.ode.expert({
                    type: 'SelfAttention',
                    inputShape,
                    projection: this.headDim
                })
                return expert.model
            })
    }

    createFeedforwardExperts(inputShape) {
        return Array(this.numExperts)
            .fill(0)
            .map((_, i) => {
                const expert = this.ode.expert({
                    type: 'GatedLinearMLP',
                    inputShape,
                    innerDim: this.mlpDim,
                    activation: 'mish'
                })
                return expert.model
            })
    }

    defineSchedulers() {
        this.minLearningRate = 0.00000001
        this.maxLearningRate = 0.00022
        const steps = 1000
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.maxLearningRate,
                steps
            )
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.maxLearningRate,
                weightDecay: 0.1
            })
        ]
    }
}

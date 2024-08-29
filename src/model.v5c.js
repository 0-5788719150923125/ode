import ODE from './model.v2.js'

/**
 * A maybe-sparse mixture of depths.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        const defaults = {
            layers: 3,
            units: 256,
            routerDim: 1024,
            headDim: 2048,
            mlpDim: 1024,
            capacity: 0.25,
            learningRate: 2e-4,
            minLearningRate: 1e-6,
            cosineSteps: 2048
        }
        super({ ...defaults, ...config })
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-consistent-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.config.units,
            embeddingsInitializer: 'glorotUniform'
        })

        const encoding = this.ode.layers.SinusoidalPositionalEncoding()

        let outputs = encoding.apply(embeddings.apply(inputs))

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .MixtureOfDepths({
                    routerDim: this.config.routerDim,
                    activation: 'gelu_new',
                    capacity: this.config.capacity,
                    experts: [
                        this.ode.expert({
                            type: 'SelfAttention',
                            inputShape: outputs.shape,
                            projection: this.config.headDim
                        }),
                        this.ode.expert({
                            type: 'GatedLinearMLP',
                            inputShape: outputs.shape,
                            hiddenDim: this.config.mlpDim,
                            activation: 'gelu_new'
                        })
                    ]
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay
            })
        ]
    }
}

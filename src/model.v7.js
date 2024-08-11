import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpenDoorExperiment extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 4
        this.units = config.units || 256
        this.numHeads = config.heads || 8
        this.queriesPerHead = config.queriesPerHead || 1
        this.headDim = config.headDim || 128
        this.numExperts = config.numExperts || 2
        this.expertDim = config.expertDim || 1024
        this.routerDim = config.routerDim || 64
        this.useBias = config.useBias || true
        this.ALiBiLength = 1024
        this.learningRate = 0.0001
        this.weightDecay = 0.00001
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-8000-clean-v1'
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

        let outputs = embeddings.apply(inputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .MultiHeadAttention({
                    numHeads: this.numHeads,
                    headDim: this.headDim,
                    queriesPerHead: this.queriesPerHead,
                    ALiBiLength: this.ALiBiLength,
                    useBias: this.useBias
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SoftMergingOfExpertsMLP({
                    activation: 'mish',
                    gateActivation: 'gelu',
                    routerActivation: 'swish',
                    numExperts: this.numExperts,
                    expertDim: this.expertDim,
                    routerDim: this.routerDim,
                    useBias: this.useBias
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Prodigy({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay
            })
        ]
    }
}

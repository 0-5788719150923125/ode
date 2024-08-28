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
        this.mlpDim = config.mlpDim || 1024
        this.useBias = config.useBias || true
        this.ALiBiLength = 1024
        this.learningRate = 0.0001
        this.weightDecay = 0.00001
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
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
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.mlpDim,
                    useBias: this.useBias
                })
                .apply(outputs)
        }

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        return [this.ode.schedulers.constantScheduler(this.learningRate)]
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay,
                useGc: true
            })
        ]
    }
}

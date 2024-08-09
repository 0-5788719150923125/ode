import ODE from './model.v2.js'

/**
 * A Frankenstein.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 4
        this.units = config.units || 256
        this.numHeads = config.numHeads || 8
        // this.queriesPerHead = config.queriesPerHead || 2
        // this.headDim = config.headDim || 128
        // this.headFeatures = config.headFeatures || 64
        this.mlpDim = config.mlpDim || 1024
        // this.useBias = config.useBias || true
        // this.ALiBiLength = 1024
        this.learningRate = 1e-4
        this.weightDecay = 1e-5
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-8000-balanced-v1'
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
                .LocalSensitiveHashingAttention({
                    numHeads: this.numHeads
                    // headDim: this.headDim,
                    // headFeatures: this.headFeatures,
                    // queriesPerHead: this.queriesPerHead,
                    // useBias: this.useBias,
                    // ALiBiLength: this.ALiBiLength
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    hiddenDim: this.mlpDim,
                    activation: 'mish',
                    gateActivation: 'swish'
                    // useBias: this.useBias
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
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay,
                useGc: true
            })
        ]
    }
}

import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpenDoorExperiment extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 12
        this.units = config.units || 64
        this.embeddings = config.embeddings || 512
        this.numHeads = config.heads || 4
        this.queriesPerHead = config.queriesPerHead || 2
        this.headDim = config.headDim || 16
        this.mlpDim = config.mlpDim || 256
        this.useBias = config.useBias || true
        this.ALiBiLength = 2048
        this.learningRate = 1.0
        this.weightDecay = 0.001
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

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.embeddings,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .dense({
                units: this.units
            })
            .apply(outputs)

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
                .MultiLayerPerceptron({
                    activation: 'laplace',
                    hiddenDim: this.mlpDim,
                    useBias: this.useBias
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

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
                weightDecay: this.weightDecay,
                biasCorrection: true
            })
        ]
    }
}

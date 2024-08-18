import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpportunisticDegenerativeExample extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 8
        this.units = config.units || 180
        this.embeddings = config.embeddings || 540
        this.numHeads = config.heads || 4
        this.queriesPerHead = config.queriesPerHead || 2
        this.headDim = config.headDim || 45
        this.mlpDim = config.mlpDim || 720
        this.useBias = config.useBias || true
        this.ALiBiLength = 2048
        this.learningRate = 1e-4
        this.minLearningRate = 1e-6
        this.weightDecay = 0.001
        this.cosineSteps = 4096
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-1024-strict-v1'
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
            .LowRankFactorization({
                units: this.units,
                rank: this.units / 2
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.numHeads,
                    headDim: this.headDim,
                    queriesPerHead: this.queriesPerHead,
                    ALiBiLength: this.ALiBiLength,
                    useBias: this.useBias
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'serf',
                    gateActivation: 'swish',
                    hiddenDim: this.mlpDim,
                    useBias: this.useBias
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: 'glorotUniform',
                useBias: this.useBias
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.learningRate,
                this.cosineSteps
            )
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lion({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay,
                useGc: true,
                adaNorm: true
            })
        ]
    }
}

import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpportunisticDegenerativeExample extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 6
        this.units = config.units || 180
        this.embeddings = config.embeddings || 540
        this.numHeads = config.heads || 4
        this.queriesPerHead = config.queriesPerHead || 2
        this.headDim = config.headDim || 45
        this.mlpDim = config.mlpDim || 1080
        this.useBias = config.useBias || true
        this.ALiBiLength = 1024
        this.learningRate = 1e-4
        this.minLearningRate = 1e-6
        this.weightDecay = 1e-5
        this.cosineSteps = 4096
        this.warmupSteps = 128
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-consistent-v1'
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
                embeddingsInitializer: this.tf.initializers.glorotUniform()
            })
            .apply(inputs)

        outputs = this.ode.layers
            .ParabolicCompression({
                units: this.units,
                numSteps: 4
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
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.mlpDim,
                    useBias: this.useBias
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: this.tf.initializers.glorotUniform()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineLossFunctions() {
        this.lossFunctions = [
            {
                function: this.ode.losses.softmaxCrossEntropy,
                smoothing: 0.0001,
                reduction: this.tf.Reduction.MEAN
            }
        ]
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.minLearningRate,
                this.learningRate,
                this.cosineSteps,
                this.warmupSteps
            )
        ]
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.Lamb({
                learningRate: this.learningRate,
                weightDecay: this.weightDecay
                // rectify: true
            })
        ]
    }
}

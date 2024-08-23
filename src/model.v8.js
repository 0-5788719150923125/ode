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
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-balanced-v1'
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

        const stepSize = Math.floor(this.units / this.layers)

        outputs = this.ode.layers
            .ParabolicCompression({
                units: stepSize,
                numSteps: 3
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            const memory = this.ode.layers
                .FastAssociativeMemory({
                    activation: 'gelu',
                    steps: 3,
                    learningRate: 1e-3,
                    decayRate: 0.9
                })
                .apply(outputs)

            const context = this.ode.layers
                .concatenate()
                .apply([outputs, memory])

            outputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.numHeads,
                    headDim: this.headDim,
                    queriesPerHead: this.queriesPerHead,
                    ALiBiLength: this.ALiBiLength,
                    useBias: this.useBias
                })
                .apply(context)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.mlpDim,
                    useBias: this.useBias
                })
                .apply(outputs)

            outputs = this.ode.layers
                .dense({
                    units: this.units / 2,
                    kernelInitializer: 'glorotNormal'
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: 'glorotUniform'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineLossFunctions() {
        this.lossFunctions = [
            {
                function: this.ode.losses.softmaxCrossEntropy,
                weights: null,
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

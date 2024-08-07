import ODE from './model.v2.js'

/**
 * A Frankenstein.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.units = config.units || 256
        this.embeddings = config.embeddings || 384
        this.layers = [
            {
                outputDim: 288,
                numHeads: 8,
                headDim: 128,
                headFeatures: 64,
                queriesPerHead: 2,
                mlpDim: 1024
            },
            {
                outputDim: 320,
                numHeads: 8,
                headDim: 128,
                headFeatures: 64,
                queriesPerHead: 2,
                mlpDim: 1024
            },
            {
                outputDim: 352,
                numHeads: 8,
                headDim: 128,
                headFeatures: 64,
                queriesPerHead: 2,
                mlpDim: 1024
            },
            {
                outputDim: this.embeddings,
                numHeads: 8,
                headDim: 128,
                headFeatures: 64,
                queriesPerHead: 2,
                mlpDim: 1024
            }
        ]
        this.reductionSteps = config.reductionSteps || 4
        this.ALiBiLength = 1024
        this.learningRate = 0.0001
        this.weightDecay = 0.00001
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
            outputDim: this.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        let outputs = embeddings.apply(inputs)

        outputs = this.ode.layers
            .ParabolicCompression({
                numSteps: this.reductionSteps,
                outputDim: this.units
            })
            .apply(outputs)

        for (const layer of this.layers) {
            outputs = this.ode.layers
                .ProjectedFeatureAttention({
                    numHeads: layer.numHeads,
                    headDim: layer.headDim,
                    headFeatures: layer.headFeatures,
                    queriesPerHead: layer.queriesPerHead,
                    outputDim: layer.outputDim,
                    ALiBiLength: this.ALiBiLength
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    hiddenDim: layer.mlpDim,
                    activation: 'mish',
                    gateActivation: 'swish'
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

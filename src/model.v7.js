import ODE from './model.v2.js'

/**
 * A Frankenstein.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.units = config.units || 100
        this.embeddings = config.embeddings || 200
        this.layers = [
            {
                outputDim: this.units,
                numHeads: 3,
                headDim: 600,
                headFeatures: 100,
                queriesPerHead: 2,
                mlpDim: 800
            },
            {
                outputDim: 150,
                numHeads: 3,
                headDim: 600,
                headFeatures: 100,
                queriesPerHead: 2,
                mlpDim: 1000
            },
            {
                outputDim: this.embeddings,
                numHeads: 3,
                headDim: 600,
                headFeatures: 100,
                queriesPerHead: 2,
                mlpDim: 1200
            }
        ]
        this.reductionSteps = config.reductionSteps || 4
        this.ALiBiLength = 1024
        this.learningRate = 0.0001
        this.minLearningRate = 0.00000001
        this.weightDecay = 0.001
        this.cosineSteps = 1024
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
                    innerDim: layer.mlpDim,
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
                weightDecay: this.weightDecay
            })
        ]
    }
}

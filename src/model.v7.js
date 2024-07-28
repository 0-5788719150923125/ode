import ODE from './model.v2.js'

/**
 * A Frankenstein.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super(config)
        this.units = config.units || 64
        this.embeddings = config.embeddings || 256
        this.layers = [
            {
                outputDim: 128,
                numHeads: 3,
                headDim: 64,
                headFeatures: 23,
                queriesPerHead: 1,
                mlpDim: 512
            },
            {
                outputDim: 192,
                numHeads: 4,
                headDim: 128,
                headFeatures: 64,
                queriesPerHead: 2,
                mlpDim: 768
            },
            {
                outputDim: this.embeddings,
                numHeads: 8,
                headDim: 256,
                headFeatures: 96,
                queriesPerHead: 2,
                mlpDim: 1024
            }
        ]
        this.reductionSteps = config.reductionSteps || 6
        this.learningRate = 0.0001
        this.minLearningRate = 0.00000001
        this.weightDecay = 0.001
        this.cosineSteps = 1024
        this.ALiBiLength = 1024
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
                    ALiBiLength: this.ALiBiLength,
                    outputDim: layer.outputDim
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

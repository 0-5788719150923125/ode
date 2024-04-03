import OriginalDecoderEngine from './model.v3.js'

/**
 * A mixture of experts.
 * @extends OriginalDecoderEngine
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 4
        this.heads = 8
        this.units = 256
        this.innerDim = this.units * 4
        this.epsilon = 1e-5
        this.alpha = 0.22
        this.topK = 2
        this.loadBalancing = 1.0
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'mistralai/Mistral-7B-v0.1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                blockSize: this.config.contextLength,
                units: this.units
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    experts: this.createExperts(),
                    units: this.units,
                    innerDim: this.innerDim,
                    topK: this.topK,
                    loadBalancing: this.loadBalancing
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    experts: this.createExperts(),
                    units: this.units,
                    innerDim: this.innerDim,
                    topK: this.topK,
                    loadBalancing: this.loadBalancing
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    createExperts() {
        return [
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.innerDim,
                heads: this.heads,
                epsilon: this.epsilon,
                activation: 'swish'
            }),
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.innerDim * 2,
                heads: this.heads / 2,
                epsilon: this.epsilon,
                activation: 'gelu'
            }),
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.innerDim / 2,
                heads: this.heads * 2,
                epsilon: this.epsilon,
                activation: 'tanh'
            })
        ]
    }

    defineOptimizers() {
        this.learningRate = 1.0
        this.optimizers = [
            this.ode.optimizers.Prodigy({
                learningRate: this.learningRate,
                weightDecay: 0.1,
                biasCorrection: true
            })
        ]
    }

    defineSchedulers() {
        this.schedulers = [
            this.ode.schedulers.constantScheduler(this.learningRate)
        ]
    }

    // defineLossFunctions() {
    //     this.lossFunctions = [
    //         {
    //             function: this.ode.losses.categoricalFocalCrossEntropy,
    //             alpha: 0.44,
    //             gamma: 2.3,
    //             fromLogits: true
    //         }
    //     ]
    // }
}

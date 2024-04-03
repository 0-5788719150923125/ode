import OriginalDecoderEngine from './model.v3.js'
import { randomString } from './utils.js'

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
        this.numExperts = 3
        this.topK = 2
        this.loadBalancing = 1.0
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'mistralai/Mistral-7B-v0.1'
        })
    }

    defineBuild() {
        const inputs = this.tf.input({
            name: `inn-${randomString()}`,
            shape: [null]
        })

        let outputs = this.tf.layers
            .embedding({
                name: `emb-${randomString()}`,
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
                    experts: new Array(this.numExperts).fill(
                        this.ode.layers.SynthesizerAttention({
                            units: this.units,
                            blockSize: this.config.contextLength,
                            heads: this.heads,
                            epsilon: this.epsilon,
                            activation: this.tf.leakyRelu,
                            alpha: this.alpha
                        })
                    ),
                    units: this.units,
                    innerDim: this.innerDim,
                    topK: this.topK,
                    loadBalancing: this.loadBalancing
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SparseMixtureOfExperts({
                    experts: new Array(this.numExperts).fill(
                        this.ode.layers.MultiLayerPerceptron({
                            units: this.units,
                            innerDim: this.innerDim,
                            heads: this.heads,
                            epsilon: this.epsilon,
                            activation: 'swish'
                        })
                    ),
                    units: this.units,
                    innerDim: this.innerDim,
                    topK: this.topK,
                    loadBalancing: this.loadBalancing
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .dense({
                name: `out-${randomString()}`,
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
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
}

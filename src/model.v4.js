import OriginalDecoderEngine from './model.v3.js'
import { randomString } from './utils.js'

/**
 * A small transformer with synthetic attention weights and rotary positional embeddings.
 * @extends OriginalDecoderEngine
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 3
        this.heads = 4
        this.units = 128
        this.innerDim = this.units * 3
        this.epsilon = 1e-6
        this.experts = []
    }

    async defineTokenizer() {
        await super.defineTokenizer({
            model: 'mistralai/Mistral-7B-v0.1'
        })
    }

    defineBuild() {
        const inputs = this.tf.input({
            name: `in1-${randomString()}`,
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

        this.experts = [
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
                heads: this.heads,
                epsilon: this.epsilon,
                activation: 'softsign'
            }),
            this.ode.layers.MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units / 2,
                heads: this.heads * 4,
                epsilon: this.epsilon,
                activation: 'linear'
            })
        ]

        const gate = this.ode.layers.ControlGate({
            experts: this.experts,
            units: this.units * 2
        })

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SynthesizerAttention({
                    units: this.units,
                    blockSize: this.config.contextLength,
                    heads: this.heads,
                    epsilon: this.epsilon,
                    activation: this.tf.leakyRelu,
                    alpha: this.alpha
                })
                .apply(outputs)

            outputs = gate.apply(outputs)
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
}

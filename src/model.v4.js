import OriginalDecoderEngine from './model.v3.js'

/**
 * A small transformer with synthetic attention weights and rotary positional embeddings.
 * @extends OriginalDecoderEngine
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 6
        this.heads = 8
        this.units = 256
        this.innerDim = this.units * 3
        this.epsilon = 1e-6
        this.operations = 23
        this.compressionFactor = 4
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })

        let outputs = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.ode.layers
            .RotaryPositionalEncoding({
                seqLen: this.config.contextLength,
                units: this.units
            })
            .apply(outputs)

        const compressor = this.ode.layers.CompressorHead({
            operations: this.operations,
            compressionFactor: this.compressionFactor
        })

        outputs = compressor.apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SynthesizerAttention({
                    units: this.units,
                    blockSize:
                        this.config.contextLength / this.compressionFactor,
                    heads: this.heads,
                    epsilon: this.epsilon,
                    activation: this.tf.leakyRelu
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.innerDim,
                    heads: this.heads,
                    epsilon: this.epsilon,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = compressor.apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

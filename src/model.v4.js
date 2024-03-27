import OriginalDecoderEngine from './model.v3.js'

/**
 * A small transformer with synthetic attention weights and sinusoidal position embeddings.
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
        this.compressionFactor = 2
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })

        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const compressed = this.ode.layers
            .CompressedEmbeddings({
                compressionFactor: this.compressionFactor,
                poolingType: 'avg'
            })
            .apply(embeddings)

        let outputs = this.ode.layers
            .RotaryPositionalEmbedding({
                seqLen: this.config.contextLength / this.compressionFactor,
                units: this.units
            })
            .apply(compressed)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .SynthesizerAttention({
                    blockSize:
                        this.config.contextLength / this.compressionFactor,
                    units: this.units,
                    heads: this.heads,
                    bias: false,
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

        outputs = this.ode.layers
            .ConvolutionalExpansionLayer({
                seqLen: this.config.contextLength,
                units: this.units
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

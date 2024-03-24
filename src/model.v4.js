import OmnipotentDiabolicalErudite from './model.v3.js'
import PretrainedTokenizer from './tokenizers.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends ModelBase
 */
export default class OriginalDecoderEngine extends OmnipotentDiabolicalErudite {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.dropout = 0.1
    }

    trainTokenizer() {
        this.tokenizer = new PretrainedTokenizer()
    }

    build() {
        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.customLayers.Range().apply(inputs)

        const positionalEmbeddings = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        let outputs = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        outputs = this.tf.layers
            .dropout({
                name: 'dropout',
                rate: this.dropout
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.customLayers
                .CausalSelfAttention({
                    blockSize: this.config.contextLength,
                    units: this.units,
                    numHeads: this.numHeads,
                    dropout: this.dropout,
                    bias: false
                })
                .apply(outputs)

            outputs = this.customLayers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.innerDim,
                    numHeads: this.numHeads,
                    dropout: this.dropout,
                    activation: 'gelu'
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                name: 'head/ln',
                epsilon: 1e-5
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                name: 'head',
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    async generate(prompt, temperature = 0.7, length = 20) {
        return await generateText.call(this, prompt, temperature, length)
    }
}

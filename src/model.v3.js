import OmnipotentDiabolicalErudite from './model.v2.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends OmnipotentDiabolicalErudite
 */
export default class OriginalDecoderEngine extends OmnipotentDiabolicalErudite {
    constructor(config) {
        super(config)
        this.layers = 4
        this.heads = 8
        this.units = 256
        this.dropout = 0.1
        this.epsilon = 1e-5
    }

    defineTokenizer() {
        this.tokenizer = this.ode.tokenizers.PretrainedTokenizer()
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

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

        outputs = this.tf.layers
            .layerNormalization({
                name: 'emb/ln',
                epsilon: this.epsilon
            })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .CausalSelfAttention({
                    blockSize: this.config.contextLength,
                    units: this.units,
                    heads: this.heads,
                    dropout: this.dropout,
                    epsilon: this.epsilon,
                    bias: false
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.innerDim,
                    heads: this.heads,
                    dropout: this.dropout,
                    epsilon: this.epsilon,
                    activation: 'gelu'
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .dense({
                name: 'head',
                units: this.tokenizer.getLength()
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

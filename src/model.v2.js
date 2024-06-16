import ODE from './model.v1.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends ODE
 */
export default class OriginalDecoderEncoder extends ODE {
    constructor(config) {
        super(config)
        this.autoregressive = true
        this.layers = config.layers || 4
        this.heads = config.heads || 8
        this.units = config.units || 128
        this.dropout = config.dropout || 0.1
        this.epsilon = config.epsilon || 1e-5
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.XenovaTokenizer({
            model: config?.model || 'openai-community/gpt2'
        })
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
                inputDim: this.contextLength,
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
                    blockSize: this.contextLength,
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

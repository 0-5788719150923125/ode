import ODE from './model.v1.js'

/**
 * A GPT-2 clone with causal attention and learned position embeddings.
 * @extends ODE
 */
export default class OriginalDecoderEncoder extends ODE {
    constructor(config) {
        const defaults = {
            layers: 4,
            units: 128,
            numHeads: 8,
            mlpDim: 512,
            dropout: 0.1,
            epsilon: 1e-5
        }
        super({ ...defaults, ...config })
        this.labels = 'multiLabel'
    }

    defineTokenizer(config) {
        return this.ode.tokenizers.XenovaTokenizer({
            model: config?.model || 'openai-community/gpt2'
        })
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.config.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

        const positionalEmbeddings = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.contextLength,
                outputDim: this.config.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        let outputs = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        outputs = this.tf.layers
            .dropout({
                name: 'dropout',
                rate: this.config.dropout
            })
            .apply(outputs)

        outputs = this.tf.layers
            .layerNormalization({
                name: 'emb/ln',
                epsilon: this.config.epsilon
            })
            .apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .GPT2Attention({
                    blockSize: this.contextLength,
                    units: this.config.units,
                    heads: this.config.numHeads,
                    dropout: this.config.dropout,
                    epsilon: this.config.epsilon,
                    bias: false
                })
                .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.config.units,
                    hiddenDim: this.config.mlpDim,
                    heads: this.config.numHeads,
                    dropout: this.config.dropout,
                    epsilon: this.config.epsilon,
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

        return this.tf.model({ inputs, outputs })
    }
}

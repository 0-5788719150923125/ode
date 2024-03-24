import OriginalDecoderEngine from './model.v4.js'
import {
    CausalSelfAttention,
    CausalAttentionLayer,
    MultiHeadAttention,
    Range,
    SinusoidalPositionalEncoding,
    MultiLayerPerceptron
} from './layers.js'
import PretrainedTokenizer from './tokenizers.js'

/**
 * A small transformer with multi-head attention and sinusoidal position embeddings.
 * @extends ModelBase
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.innerDim = this.units * 4
    }

    build() {
        const inputs = this.tf.input({ shape: [null] })

        const embeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = new Range().apply(inputs)

        const encoding = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        // const encoding = new SinusoidalPositionalEncoding({
        //     embeddingDim: this.units,
        //     reverse: false
        // }).apply(range)

        let outputs = this.tf.layers.add().apply([embeddings, encoding])

        for (let i = 0; i < this.layers; i++) {
            outputs = new CausalSelfAttention({
                blockSize: this.config.contextLength,
                nEmbd: this.units,
                nHead: this.numHeads,
                dropout: this.dropout,
                bias: false
            }).apply(outputs)

            outputs = new MultiLayerPerceptron({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                activation: 'swish'
            }).apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                epsilon: 1e-5
            })
            .apply(outputs)

        const head = this.tf.layers.dense({
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        outputs = head.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

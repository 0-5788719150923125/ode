import OriginalDecoderEngine from './model.v4.js'
import {
    MultiHeadAttention,
    Range,
    SinusoidalPositionalEncoding,
    TransformerBlock
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
        super.build()

        const inputs = this.tf.input({ shape: [null] })

        const tokenEmbeddings = this.tf.layers
            .embedding({
                name: 'wte',
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = new Range().apply(inputs)

        const positionalEmbeddings = this.tf.layers
            .embedding({
                name: 'wpe',
                inputDim: this.config.contextLength,
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(range)

        // const positionalEmbeddings = new SinusoidalPositionalEncoding({
        //     embeddingDim: this.units,
        //     reverse: false
        // }).apply(range)

        let outputs = this.tf.layers
            .add()
            .apply([tokenEmbeddings, positionalEmbeddings])

        for (let i = 0; i < this.layers; i++) {
            const attention = new MultiHeadAttention({
                numHeads: this.numHeads,
                units: this.units
            })
            outputs = attention.apply(outputs)
            const decoder = new TransformerBlock({
                units: this.units,
                innerDim: this.innerDim,
                numHeads: this.numHeads,
                activation: 'swish'
            })
            outputs = decoder.apply(outputs)
        }

        const head = this.tf.layers.dense({
            inputDim: this.units,
            units: this.tokenizer.getLength(),
            activation: 'linear'
        })

        outputs = head.apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    // async compile() {
    //     this.lossFunctions = [this.tf.losses.softmaxCrossEntropy]
    //     this.model.compile({
    //         optimizer: this.tf.train.adam(
    //             this.config.learningRate || 1e-3,
    //             this.config.beta1 || 0.9,
    //             this.config.beta2 || 0.999,
    //             this.config.epsilon || 1e-7
    //         ),
    //         loss: this.lossFunctions
    //     })
    // }
}

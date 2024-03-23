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
        this.units = 512
        this.innerDim = this.units * 4
    }

    setupTokenizer(vocabSize = 16666, numIterations = 500_000_000) {
        this.tokenizer = new PretrainedTokenizer()
    }

    build() {
        super.build()

        const inputs = this.tf.input({ shape: [null] })
        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform',
                maskZero: true
            })
            .apply(inputs)

        const range = new Range().apply(inputs)

        const encoding = new SinusoidalPositionalEncoding({
            embeddingDim: this.units,
            maxSeqLength: this.config.contextLength
        }).apply(range)

        let outputs = this.tf.layers.add().apply([embeddings, encoding])

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

    async compile() {
        this.lossFunctions = [this.tf.losses.softmaxCrossEntropy]
        this.model.compile({
            optimizer: this.tf.train.adam(
                this.config.learningRate || 1e-3,
                this.config.beta1 || 0.9,
                this.config.beta2 || 0.999,
                this.config.epsilon || 1e-7
            ),
            loss: this.lossFunctions
        })
    }
}

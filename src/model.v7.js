import OpportunisticDialogueEncoder from './model.v4.js'

/**
 * A binary compression model. It's fatally-flawed currently.
 * @extends OpportunisticDialogueEncoder
 */
export default class OrthogonallyDecayingExponent extends OpportunisticDialogueEncoder {
    constructor(config) {
        super(config)
        this.layers = 6
        this.heads = 8
        this.units = 512
        this.innerDim = this.units * 4
        this.epsilon = 1e-5
        this.alpha = 0.22
    }

    async defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.CompressedBinaryTokenizer({
            minLength: 1,
            maxLength: 12,
            maxVocabLength: 8888,
            corpus: this.config.corpus
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
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
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

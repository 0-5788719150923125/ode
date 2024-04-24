import ODE from './model.v4.js'

/**
 * For vision-based language modeling.
 * @extends ODE
 */
export default class ObjectivelyDumbExample extends ODE {
    constructor(config) {
        super(config)
        this.units = 256
        this.sourceFormat = 'image'
        this.imageSize = 512
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.Text2Image({
            imageSize: this.imageSize
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [this.imageSize, this.imageSize, 1]
        })

        let outputs = inputs

        const filters = [32, 64, 128, 256, 512]

        for (const filter of filters) {
            outputs = this.tf.layers
                .conv2d({
                    filters: filter,
                    kernelSize: 3,
                    activation: 'swish',
                    padding: 'same'
                })
                .apply(outputs)

            outputs = this.tf.layers
                .maxPooling2d({
                    poolSize: [2, 2],
                    strides: [2, 2]
                })
                .apply(outputs)
        }

        // Flatten the output
        outputs = this.tf.layers.flatten().apply(outputs)

        // Dense layers
        outputs = this.tf.layers
            .dense({
                units: this.units,
                activation: 'mish'
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.config.contextLength * this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        // Reshape the output to match the desired shape
        outputs = this.tf.layers
            .reshape({
                targetShape: [
                    this.config.contextLength,
                    this.tokenizer.getLength()
                ]
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

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
        this.encoderLayers = config.encoderLayers || 6
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

        for (let i = 0; i < this.encoderLayers; i++) {
            outputs = this.tf.layers
                .conv2d({
                    filters: this.units,
                    kernelSize: 3,
                    activation: 'swish',
                    padding: 'same'
                })
                .apply(outputs)

            outputs = this.tf.layers
                .conv2d({
                    filters: this.units,
                    kernelSize: 1,
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

        // Global average pooling
        outputs = this.tf.layers
            .globalAveragePooling2d({ dataFormat: 'channelsLast' })
            .apply(outputs)

        // Repeat the feature vector to create the time steps dimension
        outputs = this.tf.layers
            .repeatVector({
                n: this.config.contextLength
            })
            .apply(outputs)

        // Dense layer to project the output to the token space
        outputs = this.tf.layers
            .timeDistributed({
                layer: this.tf.layers.dense({
                    units: this.tokenizer.getLength(),
                    activation: 'linear'
                })
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

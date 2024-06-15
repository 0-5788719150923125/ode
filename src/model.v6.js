import ODE from './model.v4.js'

/**
 * For vision-based language modeling.
 * @extends ODE
 */
export default class ObjectivelyDumbExperiment extends ODE {
    constructor(config) {
        super(config)
        this.units = 512
        this.sourceFormat = 'image'
        this.imageSize = 256
    }

    defineTokenizer(config) {
        this.tokenizer = this.ode.tokenizers.Text2Image({
            imageSize: this.imageSize
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [-1, this.contextLength, this.imageSize, this.imageSize, 1]
        })

        let outputs = inputs

        const filters = [64, 64, 64]
        for (const filter of filters) {
            outputs = this.tf.layers
                .timeDistributed({
                    layer: this.tf.layers.conv2d({
                        filters: filter,
                        kernelSize: 3,
                        activation: 'swish',
                        padding: 'same',
                        dilationRate: 1
                    })
                })
                .apply(outputs)

            // outputs = this.tf.layers
            //     .timeDistributed({
            //         layer: this.tf.layers.maxPooling2d({
            //             poolSize: [2, 2],
            //             strides: [2, 2]
            //         })
            //     })
            //     .apply(outputs)

            outputs = this.tf.layers
                .reshape({
                    targetShape: [
                        this.contextLength,
                        outputs.shape[2],
                        outputs.shape[3]
                        // outputs.shape[4]
                    ]
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .timeDistributed({
                layer: this.tf.layers.globalAveragePooling2d({
                    dataFormat: 'channelsLast'
                })
            })
            .apply(outputs)

        outputs = this.tf.layers
            .reshape({
                targetShape: [this.contextLength, filters[filters.length - 1]]
            })
            .apply(outputs)

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }
}

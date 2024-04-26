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
        this.imageSize = 256
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

        const filters = [256, 128, 128, 64, 64, 64]

        for (const i in filters) {
            outputs = this.tf.layers
                .conv2d({
                    filters: filters[i],
                    kernelSize: 3,
                    activation: 'swish',
                    padding: 'same',
                    dilationRate: 1
                })
                .apply(outputs)

            if (i % 3 !== 0) {
                outputs = this.tf.layers
                    .maxPooling2d({
                        poolSize: [2, 2],
                        strides: [2, 2]
                    })
                    .apply(outputs)
            }

            console.log(outputs.shape)
        }

        outputs = this.tf.layers
            .reshape({
                targetShape: [
                    1,
                    outputs.shape[1] * outputs.shape[2] * outputs.shape[3]
                ]
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.units,
                activation: 'swish'
            })
            .apply(outputs)

        outputs = this.ode.layers
            .SelfAttention({
                units: this.units,
                projection: this.units * 4
            })
            .apply(outputs)

        outputs = this.ode.layers
            .MultiLayerPerceptron({
                units: this.units,
                innerDim: this.units * 4,
                activation: 'mish'
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.config.contextLength,
                activation: 'swish'
            })
            .apply(outputs)

        outputs = this.tf.layers
            .reshape({
                targetShape: [this.config.contextLength, 1]
            })
            .apply(outputs)

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

// import ODE from './model.v4.js'

// /**
//  * For vision-based language modeling.
//  * @extends ODE
//  */
// export default class ObjectivelyDumbExample extends ODE {
//     constructor(config) {
//         super(config)
//         this.units = 64
//         this.sourceFormat = 'image'
//         this.imageSize = 512
//         this.encoderLayers = config.encoderLayers || 5
//     }

//     defineTokenizer(config) {
//         this.tokenizer = this.ode.tokenizers.Text2Image({
//             imageSize: this.imageSize
//         })
//     }

//     defineBuild() {
//         const inputs = this.ode.layers.input({
//             shape: [this.imageSize, this.imageSize, 1]
//         })

//         let outputs = inputs

//         for (let i = 0; i < this.encoderLayers; i++) {
//             outputs = this.tf.layers
//                 .conv2d({
//                     filters: this.units,
//                     kernelSize: 3,
//                     activation: 'swish',
//                     padding: 'same',
//                     dilationRate: 1
//                 })
//                 .apply(outputs)

//             outputs = this.tf.layers
//                 .conv2d({
//                     filters: this.units,
//                     kernelSize: 1,
//                     activation: 'swish',
//                     padding: 'same'
//                 })
//                 .apply(outputs)

//             // outputs = this.tf.layers
//             //     .maxPooling2d({
//             //         poolSize: [2, 2],
//             //         strides: [2, 2]
//             //     })
//             //     .apply(outputs)

//             // const originalShape = outputs.shape
//             // outputs = this.tf.layers
//             //     .reshape({
//             //         targetShape: [
//             //             originalShape[2] * originalShape[3],
//             //             originalShape[1]
//             //         ]
//             //     })
//             //     .apply(outputs)

//             // outputs = this.ode.layers
//             //     .SelfAttention({
//             //         units: originalShape[3],
//             //         projection: originalShape[3] * 4
//             //     })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .reshape({
//             //         targetShape: [
//             //             originalShape[1],
//             //             originalShape[2],
//             //             originalShape[3]
//             //         ]
//             //     })
//             //     .apply(outputs)

//             console.log(outputs.shape)
//         }

//         outputs = this.tf.layers
//             .reshape({
//                 targetShape: [
//                     1,
//                     outputs.shape[1] * outputs.shape[2] * outputs.shape[3]
//                 ]
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .dense({
//                 units: this.config.contextLength,
//                 activation: 'swish'
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .reshape({
//                 targetShape: [this.config.contextLength, 1]
//             })
//             .apply(outputs)

//         // outputs = this.tf.layers
//         //     .dense({
//         //         units: this.units,
//         //         activation: 'swish'
//         //     })
//         //     .apply(outputs)

//         // outputs = this.tf.layers
//         //     .dense({
//         //         units: this.config.contextLength,
//         //         activation: 'swish'
//         //     })
//         //     .apply(outputs)

//         outputs = this.tf.layers
//             .timeDistributed({
//                 layer: this.tf.layers.dense({
//                     units: this.tokenizer.getLength(),
//                     activation: 'linear'
//                 })
//             })
//             .apply(outputs)

//         this.model = this.tf.model({ inputs, outputs })
//     }
// }

// import ODE from './model.v4.js'

// /**
//  * For vision-based language modeling.
//  * @extends ODE
//  */
// export default class ObjectivelyDumbExample extends ODE {
//     constructor(config) {
//         super(config)
//         this.units = 256
//         this.sourceFormat = 'image'
//         this.imageSize = 512
//         this.encoderLayers = config.encoderLayers || 6
//     }

//     defineTokenizer(config) {
//         this.tokenizer = this.ode.tokenizers.Text2Image({
//             imageSize: this.imageSize
//         })
//     }

//     defineBuild() {
//         const inputs = this.ode.layers.input({
//             shape: [this.imageSize, this.imageSize, 1]
//         })

//         let outputs = inputs

//         for (let i = 0; i < this.encoderLayers; i++) {
//             const stride = i % 2 === 0 ? 1 : 2
//             outputs = this.tf.layers
//                 .conv2d({
//                     filters: this.units,
//                     kernelSize: 3,
//                     activation: 'swish',
//                     padding: 'same',
//                     strides: stride
//                 })
//                 .apply(outputs)

//             outputs = this.tf.layers
//                 .globalAveragePooling2d({ dataFormat: 'channelsLast' })
//                 .apply(outputs)

//             outputs = this.tf.layers
//                 .reshape({
//                     targetShape: [1, ...outputs.shape.slice(1)]
//                 })
//                 .apply(outputs)

//             outputs = this.ode.layers
//                 .SelfAttention({
//                     units: this.units,
//                     projection: this.units * 4
//                 })
//                 .apply(outputs)

//             outputs = this.ode.layers
//                 .GatedLinearUnit({
//                     units: this.units,
//                     innerDim: this.units * 4,
//                     activation: 'swish'
//                 })
//                 .apply(outputs)

//             outputs = this.tf.layers
//                 .reshape({
//                     targetShape: [1, 1, this.units]
//                 })
//                 .apply(outputs)
//         }

//         outputs = this.tf.layers
//             .globalAveragePooling2d({ dataFormat: 'channelsLast' })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .dense({
//                 units: this.config.contextLength,
//                 activation: 'swish'
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .reshape({
//                 targetShape: [this.config.contextLength, 1]
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .dense({
//                 units: this.config.contextLength,
//                 activation: 'swish'
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .timeDistributed({
//                 layer: this.tf.layers.dense({
//                     units: this.tokenizer.getLength(),
//                     activation: 'linear'
//                 })
//             })
//             .apply(outputs)

//         this.model = this.tf.model({ inputs, outputs })
//     }
// }

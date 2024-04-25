import ODE from './model.v4.js'

/**
 * For vision-based language modeling.
 * @extends ODE
 */
export default class ObjectivelyDumbExample extends ODE {
    constructor(config) {
        super(config)
        this.units = 64
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
            // console.log(outputs.shape)

            const stride = i % 2 === 0 ? 1 : 2
            outputs = this.tf.layers
                .conv2d({
                    filters: this.units,
                    kernelSize: 3,
                    activation: 'swish',
                    padding: 'same',
                    strides: stride
                })
                .apply(outputs)

            outputs = this.tf.layers
                .globalAveragePooling2d({ dataFormat: 'channelsLast' })
                .apply(outputs)
            console.log(outputs.shape)
            outputs = this.tf.layers
                .dense({
                    units: this.units,
                    activation: 'swish'
                })
                .apply(outputs)

            outputs = this.tf.layers
                .reshape({
                    targetShape: [1, ...outputs.shape.slice(1)]
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SelfAttention({
                    units: this.units,
                    projection: this.units * 4
                })
                .apply(outputs)

            outputs = this.tf.layers
                .reshape({
                    targetShape: [1, 1, this.units]
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .globalAveragePooling2d({ dataFormat: 'channelsLast' })
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

        // outputs = this.tf.layers
        //     .dense({
        //         units: this.config.contextLength,
        //         activation: 'swish'
        //     })
        //     .apply(outputs)

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
//             console.log(outputs.shape)

//             const kernelSize = 3
//             outputs = this.tf.layers
//                 .conv2d({
//                     filters: this.units,
//                     kernelSize: kernelSize,
//                     activation: 'swish',
//                     padding: 'same'
//                 })
//                 .apply(outputs)

//             const originalShape = outputs.shape

// outputs = this.tf.layers
//     .reshape({
//         targetShape: [this.units, kernelSize * kernelSize * this.units + 1]
//     })
//     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .reshape({
//             //         targetShape: [
//             //             originalShape[1],
//             //             originalShape[2],
//             //             originalShape[3]
//             //         ]
//             //     })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .reshape({
//             //         targetShape: [this.units, kernelSize, kernelSize * this.units + 1]
//             //     })
//             //     .apply(outputs)

//             // outputs = this.ode.layers
//             //     .SelfAttention({
//             //         units: this.units,
//             //         projection: this.units * 4
//             //     })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .timeDistributed({
//             //         layer: this.ode.layers.SelfAttention({
//             //             units: this.units,
//             //             projection: this.units * 4
//             //         })
//             //     })
//             //     .apply(outputs)

//             // if (i === this.encoderLayers - 1) continue

//             // outputs = this.tf.layers
//             //     .reshape({
//             //         targetShape: [1, this.imageSize, this.imageSize]
//             //     })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .averagePooling2d({ dataFormat: 'channelsLast' })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .timeDistributed({
//             //         layer: this.ode.layers.SelfAttention({
//             //             units: this.units,
//             //             projection: this.units * 4
//             //         })
//             //     })
//             //     .apply(outputs)

//             // outputs = this.tf.layers
//             //     .maxPooling2d({
//             //         poolSize: [2, 2],
//             //         strides: [2, 2]
//             //     })
//             //     .apply(outputs)
//         }

//         // // Global average pooling
//         // outputs = this.tf.layers
//         //     .globalAveragePooling2d({ dataFormat: 'channelsLast' })
//         //     .apply(outputs)

//         // // Repeat the feature vector to create the time steps dimension
//         // outputs = this.tf.layers
//         //     .repeatVector({
//         //         n: this.config.contextLength
//         //     })
//         //     .apply(outputs)

//         // outputs = this.tf.layers
//         //     .timeDistributed({
//         //         layer: this.ode.layers.SelfAttention({
//         //             units: this.units,
//         //             projection: this.units * 4
//         //         })
//         //     })
//         //     .apply(outputs)

//         // outputs = this.tf.layers
//         //     .dense({
//         //         units: this.tokenizer.getLength(),
//         //         activation: 'linear'
//         //     })
//         //     .apply(outputs)

//         outputs = this.tf.layers
//             .reshape({
//                 targetShape: [this.imageSize, this.imageSize * this.units]
//             })
//             .apply(outputs)

//         outputs = this.tf.layers
//             .dense({
//                 units: this.tokenizer.getLength(),
//                 activation: 'linear'
//             })
//             .apply(outputs)

//         // Dense layer to project the output to the token space
//         // outputs = this.tf.layers
//         //     .timeDistributed({
//         //         layer: this.tf.layers.dense({
//         //             units: this.tokenizer.getLength(),
//         //             activation: 'linear'
//         //         })
//         //     })
//         //     .apply(outputs)

//         this.model = this.tf.model({ inputs, outputs })
//     }
// }

import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
        this.numFeatures = config.numFeatures || 256
    }

    build(inputShape) {
        const units = inputShape[inputShape.length - 1]

        this.queryKernels = []
        this.keyKernels = []
        this.valueKernels = []
        this.featureMatrices = []

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels.push(
                this.addWeight(
                    `queryKernel_${i}`,
                    [units, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.keyKernels.push(
                this.addWeight(
                    `keyKernel_${i}`,
                    [units, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.valueKernels.push(
                this.addWeight(
                    `valueKernel_${i}`,
                    [units, this.headDim],
                    'float32',
                    tf.initializers.glorotUniform(),
                    tf.regularizers.l2({ l2: 0.01 })
                )
            )
            this.featureMatrices.push(
                tf.randomNormal(
                    [this.headDim, this.numFeatures],
                    0,
                    1 / Math.sqrt(this.numFeatures)
                )
            )
        }

        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.headDim * this.numHeads, units],
            'float32',
            tf.initializers.glorotUniform(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const attentionOutputs = []

            for (let i = 0; i < this.numHeads; i++) {
                const Q = this.applyDense(inputs, this.queryKernels[i].read())
                const K = this.applyDense(inputs, this.keyKernels[i].read())
                const V = this.applyDense(inputs, this.valueKernels[i].read())

                const Qp = this.applyFeatureMap(Q, this.featureMatrices[i])
                const Kp = this.applyFeatureMap(K, this.featureMatrices[i])

                const scores = tf
                    .matMul(Qp, Kp, false, true)
                    .div(tf.scalar(Math.sqrt(this.numFeatures)))
                    .add(mask)

                let weights = scores.softmax()

                const output = tf.matMul(weights, V)

                attentionOutputs.push(output)
            }

            const concatenatedOutputs = tf.concat(attentionOutputs, -1)
            let outputs = this.applyDense(
                concatenatedOutputs,
                this.outputKernel.read()
            )

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.add(inputs, outputs)

            return outputs
        })
    }

    applyFeatureMap(x, featureMatrix) {
        const projection = tf.matMul(x, featureMatrix)
        // ReLU activation for sparsity and efficiency
        return tf.relu(projection)
    }

    getWeights() {
        const weights = []

        for (let i = 0; i < this.numHeads; i++) {
            weights.push(this.queryKernels[i].read())
            weights.push(this.keyKernels[i].read())
            weights.push(this.valueKernels[i].read())
            weights.push(this.featureMatrices[i])
        }

        weights.push(this.outputKernel.read())

        return weights
    }

    setWeights(weights) {
        let index = 0

        for (let i = 0; i < this.numHeads; i++) {
            this.queryKernels[i].write(weights[index++])
            this.keyKernels[i].write(weights[index++])
            this.valueKernels[i].write(weights[index++])
            this.featureMatrices[i] = weights[index++]
        }

        this.outputKernel.write(weights[index])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numHeads: this.numHeads,
            headDim: this.headDim,
            numFeatures: this.numFeatures
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)

// import * as tf from '@tensorflow/tfjs'
// import LayerBase from './base.js'

// export default class ProjectedFeatureAttention extends LayerBase {
//     constructor(config) {
//         super(config)
//         this.numHeads = config.numHeads || 8
//         this.headDim = config.headDim || 64
//         this.numFeatures = config.numFeatures || 256
//         this.dropout = config.dropout || 0
//     }

//     build(inputShape) {
//         const units = inputShape[inputShape.length - 1]

//         this.queryKernels = []
//         this.keyKernels = []
//         this.valueKernels = []
//         this.featureMatrices = []

//         for (let i = 0; i < this.numHeads; i++) {
//             this.queryKernels.push(
//                 this.addWeight(
//                     `queryKernel_${i}`,
//                     [units, this.headDim],
//                     'float32',
//                     tf.initializers.glorotUniform(),
//                     tf.regularizers.l2({ l2: 0.01 })
//                 )
//             )
//             this.keyKernels.push(
//                 this.addWeight(
//                     `keyKernel_${i}`,
//                     [units, this.headDim],
//                     'float32',
//                     tf.initializers.glorotUniform(),
//                     tf.regularizers.l2({ l2: 0.01 })
//                 )
//             )
//             this.valueKernels.push(
//                 this.addWeight(
//                     `valueKernel_${i}`,
//                     [units, this.headDim],
//                     'float32',
//                     tf.initializers.glorotUniform(),
//                     tf.regularizers.l2({ l2: 0.01 })
//                 )
//             )
//             this.featureMatrices.push(
//                 this.addWeight(
//                     `featureMatrix_${i}`,
//                     [this.headDim, this.numFeatures],
//                     'float32',
//                     tf.initializers.glorotUniform(),
//                     tf.regularizers.l2({ l2: 0.01 })
//                 )
//             )
//             // this.randomMatrix = tf.randomNormal(
//             //     [this.headDim, this.numFeatures],
//             //     0,
//             //     1 / Math.sqrt(this.numFeatures)
//             // )
//         }

//         this.outputKernel = this.addWeight(
//             'outputKernel',
//             [this.headDim * this.numHeads, units],
//             'float32',
//             tf.initializers.glorotUniform(),
//             tf.regularizers.l2({ l2: 0.01 })
//         )
//     }

//     call(inputs, kwargs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const mask = tf.linalg
//                 .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
//                 .sub(tf.eye(inputs.shape[1]))
//                 .mul(tf.scalar(-1e9))

//             const attentionOutputs = []

//             for (let i = 0; i < this.numHeads; i++) {
//                 const Q = this.applyDense(inputs, this.queryKernels[i].read())
//                 const K = this.applyDense(inputs, this.keyKernels[i].read())
//                 const V = this.applyDense(inputs, this.valueKernels[i].read())

//                 const Qp = this.applyDense(Q, this.featureMatrices[i].read())
//                 const Kp = this.applyDense(K, this.featureMatrices[i].read())

//                 const scores = tf
//                     .matMul(Qp, Kp, false, true)
//                     .div(tf.scalar(Math.sqrt(this.numFeatures)))
//                     .add(mask)

//                 let weights = scores.softmax()

//                 weights = kwargs['training']
//                     ? tf.dropout(weights, this.dropout)
//                     : weights

//                 const output = tf.matMul(weights, V)

//                 attentionOutputs.push(output)
//             }

//             const concatenatedOutputs = tf.concat(attentionOutputs, -1)
//             let outputs = this.applyDense(
//                 concatenatedOutputs,
//                 this.outputKernel.read()
//             )

//             outputs = this.ops.rmsNorm(outputs)

//             outputs = tf.add(inputs, outputs)

//             outputs = kwargs['training']
//                 ? tf.dropout(outputs, this.dropout)
//                 : outputs

//             return outputs
//         })
//     }

//     applyFeatureMap(x) {
//         const projection = tf.matMul(x, this.randomMatrix)
//         // ReLU activation for sparsity and efficiency
//         return tf.relu(projection)
//     }

//     getWeights() {
//         const weights = []

//         for (let i = 0; i < this.numHeads; i++) {
//             weights.push(this.queryKernels[i].read())
//             weights.push(this.keyKernels[i].read())
//             weights.push(this.valueKernels[i].read())
//             weights.push(this.featureMatrices[i].read())
//         }

//         weights.push(this.outputKernel.read())

//         return weights
//     }

//     setWeights(weights) {
//         let index = 0

//         for (let i = 0; i < this.numHeads; i++) {
//             this.queryKernels[i].write(weights[index++])
//             this.keyKernels[i].write(weights[index++])
//             this.valueKernels[i].write(weights[index++])
//             this.featureMatrices[i].write(weights[index++])
//         }

//         this.outputKernel.write(weights[index])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             numHeads: this.numHeads,
//             headDim: this.headDim,
//             numFeatures: this.numFeatures,
//             dropout: this.dropout
//         }
//     }
// }

// tf.serialization.registerClass(ProjectedFeatureAttention)

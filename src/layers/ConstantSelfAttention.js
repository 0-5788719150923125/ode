import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// By projecting features into a lower dimension, we can keep memory
// consumption at a constant, management level.
export default class ConstantSelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 256
        this.numFeatures = config.numFeatures || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )

        this.featureMatrix = this.addWeight(
            'featureMatrix',
            [this.hiddenDim, this.numFeatures],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const Q = this.applyDense(inputs, this.queryKernel.read())
            const K = this.applyDense(inputs, this.keyKernel.read())
            const V = this.applyDense(inputs, this.valueKernel.read())

            const Qp = this.applyDense(Q, this.featureMatrix.read())
            const Kp = this.applyDense(K, this.featureMatrix.read())

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf.matMul(Qp, Kp, false, true).add(mask)

            const weights = scores.div(tf.scalar(Math.sqrt(this.numFeatures)))

            const outputs = tf.matMul(weights, V)

            const normalized = this.ops.rmsNorm(outputs)

            return tf.add(inputs, normalized)
        })
    }

    getWeights() {
        return [
            this.queryKernel.read(),
            this.keyKernel.read(),
            this.valueKernel.read(),
            this.featureMatrix.read()
        ]
    }

    setWeights(weights) {
        this.queryKernel.write(weights[0])
        this.keyKernel.write(weights[1])
        this.valueKernel.write(weights[2])
        this.featureMatrix.write(weights[3])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numFeatures: this.numFeatures
        }
    }
}

tf.serialization.registerClass(ConstantSelfAttention)

// import * as tf from '@tensorflow/tfjs'
// import LayerBase from './base.js'

// export default class ConstantSelfAttention extends LayerBase {
//     constructor(config) {
//         super(config)
//         this.hiddenDim = config.hiddenDim || 256
//         this.numFeatures = config.numFeatures || 64
//     }

//     build(inputShape) {
//         const inputDim = inputShape[inputShape.length - 1]

//         this.queryKernel = this.addWeight(
//             'queryKernel',
//             [inputDim, this.hiddenDim],
//             'float32',
//             tf.initializers.glorotUniform()
//         )

//         this.keyKernel = this.addWeight(
//             'keyKernel',
//             [inputDim, this.hiddenDim],
//             'float32',
//             tf.initializers.glorotUniform()
//         )

//         this.valueKernel = this.addWeight(
//             'valueKernel',
//             [inputDim, inputDim],
//             'float32',
//             tf.initializers.glorotUniform()
//         )
//     }

//     call(inputs) {
//         return tf.tidy(() => {
//             inputs = Array.isArray(inputs) ? inputs[0] : inputs

//             const Q = this.applyDense(inputs, this.queryKernel.read())
//             const K = this.applyDense(inputs, this.keyKernel.read())
//             const V = this.applyDense(inputs, this.valueKernel.read())

//             const Qp = this.generateRandomFeatures(Q)
//             const Kp = this.generateRandomFeatures(K)

//             const mask = tf.linalg
//                 .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
//                 .sub(tf.eye(inputs.shape[1]))
//                 .mul(tf.scalar(-1e9))

//             const scores = tf.matMul(Qp, Kp, false, true).add(mask)

//             const weights = scores.div(tf.scalar(this.numFeatures))

//             const outputs = tf.matMul(weights, V)

//             const normalized = this.ops.rmsNorm(outputs)

//             return tf.add(inputs, normalized)
//         })
//     }

//     generateRandomFeatures(inputs) {
//         const dims = inputs.shape[inputs.shape.length - 1]
//         const W = tf.randomNormal([dims, this.numFeatures])
//         const b = tf.randomUniform([this.numFeatures], 0, 2 * Math.PI)
//         const features = tf.matMul(inputs, W).add(b).cos()
//         return features
//     }

//     getWeights() {
//         return [
//             this.queryKernel.read(),
//             this.keyKernel.read(),
//             this.valueKernel.read()
//         ]
//     }

//     setWeights(weights) {
//         this.queryKernel.write(weights[0])
//         this.keyKernel.write(weights[1])
//         this.valueKernel.write(weights[2])
//     }

//     getConfig() {
//         return {
//             ...super.getConfig(),
//             hiddenDim: this.hiddenDim,
//             numFeatures: this.numFeatures
//         }
//     }
// }

// tf.serialization.registerClass(ConstantSelfAttention)

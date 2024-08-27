import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class QSparseMLP extends LayerBase {
    constructor(config) {
        super(config)
        this.innerDim = config?.innerDim || 1024
        this.units = config.units || null
        this.sparsityRatio = config?.sparsityRatio || 0.5
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]

        // Initialize dense layers for projection
        this.inProjKernel = this.addWeight(
            `inProjKernel`,
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        if (this.useBias)
            this.inProjBias = this.addWeight(
                `inProjBias`,
                [this.innerDim],
                'float32',
                tf.initializers.zeros()
            )

        this.outProjKernel = this.addWeight(
            `outProjKernel`,
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        if (this.useBias)
            this.outProjBias = this.addWeight(
                `outProjBias`,
                [this.units],
                'float32',
                tf.initializers.zeros()
            )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand projection via feedforward layer
            let outputs = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.useBias ? this.inProjBias.read() : null
            )

            outputs = this.ops.rmsNorm(outputs)

            // Apply squared ReLU activation
            const squaredReLU = tf.square(tf.relu(outputs))

            // Apply sigmoid activation
            const sigmoid = tf.sigmoid(outputs)

            // Perform element-wise multiplication (GLU)
            outputs = tf.mul(squaredReLU, sigmoid)

            // Custom top-K sparsity function with straight-through estimator
            outputs = this.sparseTopKWithSTE(outputs, this.sparsityRatio)

            outputs = tf.div(outputs, tf.norm(outputs, 2, -1, true))

            // Contract projection via feedforward layer
            outputs = this.ops.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.useBias ? this.outProjBias.read() : null
            )

            // Residual connection
            outputs = tf.add(inputs, outputs)

            return outputs
        })
    }

    sparseTopKWithSTE(inputs, sparsityRatio) {
        return tf.customGrad((inputs, save) => {
            const k = Math.floor(
                inputs.shape[inputs.shape.length - 1] * sparsityRatio
            )
            const topK = tf.topk(tf.abs(inputs), k)
            const mask = tf.greaterEqual(tf.abs(inputs), topK.values.min())
            const sparseOutputs = tf.mul(inputs, tf.cast(mask, inputs.dtype))

            save([inputs])

            return {
                value: sparseOutputs,
                gradFunc: (dy, saved) => [dy]
            }
        })(inputs)
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            sparsityRatio: this.sparsityRatio
        }
    }
}

tf.serialization.registerClass(QSparseMLP)

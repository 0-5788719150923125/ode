import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super(config)
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.activation = config?.activation || 'relu'
        this.units = config.units || null
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]

        // Initialize dense layers for projection
        this.inProjKernel = this.addWeight(
            `inProjKernel`,
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
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
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
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

            // Expand and contract projection via feedforward layers
            let outputs = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.useBias ? this.inProjBias.read() : null
            )

            outputs = this.ops.rmsNorm(outputs)

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.useBias ? this.outProjBias.read() : null
            )

            // Residual connection
            outputs = tf.add(inputs, outputs)

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            activation: this.activation
        }
    }
}

tf.serialization.registerClass(MultiLayerPerceptron)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class MultiLayerPerceptron extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config?.hiddenDim || 1024
        this.dropout = config?.dropout || 0
        this.activation = config?.activation || 'relu'
        this.units = config.units || null
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]
        this.inProjKernel = this.addWeight(
            `inProjKernel`,
            [this.units, this.hiddenDim],
            'float32',
            this.initializers.glorotNormal()
        )
        this.outProjKernel = this.addWeight(
            `outProjKernel`,
            [this.hiddenDim, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        if (this.useBias) {
            this.inProjBias = this.addWeight(
                `inProjBias`,
                [this.hiddenDim],
                'float32',
                this.initializers.zeros()
            )
            this.outProjBias = this.addWeight(
                `outProjBias`,
                [this.units],
                'float32',
                this.initializers.zeros()
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            let outputs = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias?.read()
            )

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.outProjBias?.read()
            )

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
            hiddenDim: this.hiddenDim,
            dropout: this.dropout,
            activation: this.activation
        }
    }
}

tf.serialization.registerClass(MultiLayerPerceptron)

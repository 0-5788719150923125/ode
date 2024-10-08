import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class VariableDimensionMLP extends LayerBase {
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
            'inProjKernel',
            [this.units, this.innerDim],
            'float32',
            this.initializers.glorotNormal()
        )
        this.inProjBias = this.addWeight(
            'inProjBias',
            [this.innerDim],
            'float32',
            this.initializers.zeros()
        )

        this.outProjKernel = this.addWeight(
            'outProjKernel',
            [this.innerDim, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        this.outProjBias = this.addWeight(
            'outProjBias',
            [this.units],
            'float32',
            this.initializers.zeros()
        )

        // Residual connections/skip connections are critical here
        this.residual = customLayers.ResidualConnection()
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Get the input dimensions
            const inputDim = inputs.shape[inputs.shape.length - 1]

            // Slice the weights based on the input dimensions
            const slicedInProjKernel = this.inProjKernel
                .read()
                .slice([0, 0], [inputDim, this.innerDim])
            const slicedOutProjKernel = this.outProjKernel
                .read()
                .slice([0, 0], [this.innerDim, inputDim])
            const slicedOutProjBias = this.outProjBias
                .read()
                .slice([0], [inputDim])

            // Expand and contract projection via feedforward layers
            let outputs = this.ops.applyDense(
                inputs,
                slicedInProjKernel,
                this.inProjBias.read()
            )

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                slicedOutProjKernel,
                slicedOutProjBias
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
            innerDim: this.innerDim,
            dropout: this.dropout,
            activation: this.activation
        }
    }
}

tf.serialization.registerClass(VariableDimensionMLP)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class SelfModeling extends LayerBase {
    constructor(config) {
        super(config)
        this.filters = config?.filters || 64
        this.kernelSize = config?.kernelSize || 3
        this.strides = config?.strides || 1
        this.activation = config?.activation || 'relu'
        this.units = config?.units || null
    }

    build(inputShape) {
        this.units = this.units ? this.units : inputShape[inputShape.length - 1]
        this.convKernel = this.addWeight(
            `convKernel`,
            [this.kernelSize, inputShape[inputShape.length - 1], this.filters],
            'float32',
            this.initializers.glorotNormal()
        )
        this.projKernel = this.addWeight(
            `projKernel`,
            [this.filters, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        if (this.useBias) {
            this.convBias = this.addWeight(
                `convBias`,
                [this.filters],
                'float32',
                this.initializers.zeros()
            )
            this.projBias = this.addWeight(
                `projBias`,
                [this.units],
                'float32',
                this.initializers.zeros()
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            let outputs = tf.conv1d(
                inputs,
                this.convKernel.read(),
                this.strides,
                'same'
            )
            outputs = this.convBias
                ? outputs.add(this.convBias.read())
                : outputs

            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                this.projKernel.read(),
                this.projBias?.read()
            )

            return outputs
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            filters: this.filters,
            kernelSize: this.kernelSize,
            strides: this.strides,
            activation: this.activation,
            units: this.units
        }
    }
}

tf.serialization.registerClass(SelfModeling)

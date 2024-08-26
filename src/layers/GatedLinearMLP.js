import * as tf from '@tensorflow/tfjs'
import MultiLayerPerceptron from './MultiLayerPerceptron.js'

export default class GatedLinearMLP extends MultiLayerPerceptron {
    constructor(config) {
        super(config)
        this.gateActivation = config.gateActivation || 'sigmoid'
    }

    build(inputShape) {
        super.build(inputShape)

        this.gateProjKernel = this.addWeight(
            `gateProjKernel`,
            [this.units, this.hiddenDim],
            'float32',
            tf.initializers.glorotNormal({
                seed: this.ops.getSeed()
            })
        )
        if (this.useBias)
            this.gateProjBias = this.addWeight(
                `gateProjBias`,
                [this.hiddenDim],
                'float32',
                tf.initializers.zeros()
            )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            let proj = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias?.read()
            )

            proj = this.ops.rmsNorm(proj)

            proj = tf.layers
                .activation({ activation: this.activation })
                .apply(proj)

            let gate = this.ops.applyDense(
                inputs,
                this.gateProjKernel.read(),
                this.gateProjBias?.read()
            )

            gate = tf.layers
                .activation({ activation: this.gateActivation })
                .apply(gate)

            const gatedOutput = tf.mul(proj, gate)

            let outputs = this.ops.applyDense(
                gatedOutput,
                this.outProjKernel.read(),
                this.outProjBias?.read()
            )

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
            gateActivation: this.gateActivation
        }
    }
}

tf.serialization.registerClass(GatedLinearMLP)

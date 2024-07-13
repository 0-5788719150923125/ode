import * as tf from '@tensorflow/tfjs'
import MultiLayerPerceptron from './MultiLayerPerceptron.js'

export default class GatedLinearMLP extends MultiLayerPerceptron {
    constructor(config) {
        super(config)
    }

    build(inputShape) {
        super.build(inputShape)

        this.gateProjKernel = this.addWeight(
            `gateProjKernel`,
            [this.units, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.gateProjBias = this.addWeight(
            `gateProjBias`,
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Expand and contract projection via feedforward layers
            let proj = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias.read()
            )

            proj = this.ops.rmsNorm(proj)

            proj = tf.layers
                .activation({ activation: this.activation })
                .apply(proj)

            let gate = this.ops.applyDense(
                inputs,
                this.gateProjKernel.read(),
                this.gateProjBias.read()
            )

            gate = tf.layers.activation({ activation: 'sigmoid' }).apply(gate)

            const gatedOutput = tf.mul(proj, gate)

            let outputs = this.ops.applyDense(
                gatedOutput,
                this.outProjKernel.read(),
                this.outProjBias.read()
            )

            // Residual connection
            outputs = tf.add(inputs, outputs)

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }
}

tf.serialization.registerClass(GatedLinearMLP)

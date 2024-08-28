import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

// https://github.com/rish-16/aft-pytorch/blob/main/aft_pytorch/aft_pytorch.py
export default class AttentionFreeTransformer extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units || 64
        this.hiddenDim = config.hiddenDim || 64
        this.contextLength = config.contextLength
    }

    build(inputShape) {
        this.toQ = this.addWeight(
            'toQ',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            this.initializers.glorotUniform()
        )
        this.toK = this.addWeight(
            'toK',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            this.initializers.glorotUniform()
        )
        this.toV = this.addWeight(
            'toV',
            [inputShape[inputShape.length - 1], this.hiddenDim],
            'float32',
            this.initializers.glorotUniform()
        )
        this.project = this.addWeight(
            'project',
            [this.hiddenDim, this.units],
            'float32',
            this.initializers.glorotUniform()
        )
        this.wbias = this.addWeight(
            'wbias',
            [this.contextLength, this.contextLength],
            'float32',
            this.initializers.glorotUniform()
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [B, T, _] = inputs.shape

            const Q = this.ops
                .applyDense(inputs, this.toQ.read())
                .reshape([B, T, this.hiddenDim])
            const K = this.ops
                .applyDense(inputs, this.toK.read())
                .reshape([B, T, this.hiddenDim])
            const V = this.ops
                .applyDense(inputs, this.to.read())
                .reshape([B, T, this.hiddenDim])

            const mask = tf.linalg
                .bandPart(tf.ones([inputs.shape[1], inputs.shape[1]]), 0, -1)
                .sub(tf.eye(inputs.shape[1]))
                .mul(tf.scalar(-1e9))

            const tempWbias = this.wbias
                .read()
                .slice([0, 0], [T, T])
                .expandDims(0)
                .add(mask)

            const QSig = tf.sigmoid(Q)
            const temp = tf.matMul(tf.exp(tempWbias), tf.mul(tf.exp(K), V))
            const weighted = tf.div(
                temp,
                tf.matMul(tf.exp(tempWbias), tf.exp(K))
            )

            const Yt = tf.mul(QSig, weighted)

            const outputs = this.ops.applyDense(
                Yt.reshape([B, T, this.hiddenDim]),
                this.project.read()
            )

            return tf.add(inputs, outputs)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            hiddenDim: this.hiddenDim,
            contextLength: this.contextLength
        }
    }
}

tf.serialization.registerClass(AttentionFreeTransformer)

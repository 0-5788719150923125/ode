import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/1610.06258
export default class FastAssociativeMemory extends LayerBase {
    constructor(config) {
        super(config)
        this.activation = config.activation || 'relu'
        this.numSteps = config.numSteps || 3
        this.learningRate = config.learningRate || 0.5
        this.decayRate = config.decayRate || 0.9
        this.hPrev = null
        this.hHistory = []
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.W = this.addWeight(
            'W',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.C = this.addWeight(
            'C',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.b = this.addWeight(
            'b',
            [inputDim],
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs, args) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, seqLen, features] = inputs.shape

            if (this.hHistory.length < 1) {
                this.hHistory.push(tf.keep(tf.zerosLike(inputs)))
            }

            let hPrev = this.hHistory[this.hHistory.length - 1]

            const prevSeqLen = hPrev.shape[1]
            if (prevSeqLen < seqLen) {
                const paddings = [
                    [0, 0],
                    [seqLen - prevSeqLen, 0],
                    [0, 0]
                ]
                hPrev = hPrev.pad(paddings, 1)
                this.hHistory = this.hHistory.map((h) => {
                    const hClone = h.clone()
                    h.dispose()
                    return tf.keep(hClone.pad(paddings, 1))
                })
            } else if (prevSeqLen > seqLen) {
                const paddings = [
                    [0, 0],
                    [prevSeqLen - seqLen, 0],
                    [0, 0]
                ]
                inputs = inputs.pad(paddings, 0)
            }

            let hInitial = this.ops.applyDense(
                inputs,
                this.C.read(),
                this.b.read()
            )

            hInitial = hInitial.add(this.ops.applyDense(hPrev, this.W.read()))

            let h = hInitial

            for (let s = 0; s < this.numSteps; s++) {
                const attentionTerms = this.hHistory.map((hHist, idx) => {
                    const scalarProduct = tf.sum(tf.mul(hHist, h), -1, true)

                    const weightedProduct = tf.mul(
                        scalarProduct,
                        Math.pow(this.decayRate, this.hHistory.length - idx - 1)
                    )
                    return tf.mul(weightedProduct, hHist)
                })

                const attention = tf.sum(tf.stack(attentionTerms), 0)

                const hNext = tf.add(
                    hInitial,
                    tf.mul(attention, this.learningRate)
                )

                h = this.ops.rmsNorm(hNext)

                h = tf.layers
                    .activation({ activation: this.activation })
                    .apply(h)
            }

            while (this.hHistory.length >= this.numSteps) {
                this.hHistory[0].dispose()
                this.hHistory.shift()
            }

            this.hHistory.push(tf.keep(h))

            const outputs = h.slice(
                [0, h.shape[1] - seqLen, 0],
                [batchSize, seqLen, -1]
            )

            return outputs
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            activation: this.activation,
            numSteps: this.numSteps,
            learningRate: this.learningRate,
            decayRate: this.decayRate
        }
    }
}

tf.serialization.registerClass(FastAssociativeMemory)

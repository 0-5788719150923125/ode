import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/1610.06258
export default class FastAssociativeMemory extends LayerBase {
    constructor(config) {
        super(config)
        this.activation = config.activation || 'relu'
        this.steps = config.steps || 3
        this.learningRate = config.learningRate || 1e-3
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
            tf.initializers.identity({ gain: 0.05 })
        )
        this.C = this.addWeight(
            'C',
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotNormal()
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

            if (!this.hPrev) {
                this.hPrev = tf.zerosLike(inputs)
                this.hHistory.push(tf.keep(this.hPrev.clone()))
            } else {
                const prevSeqLen = this.hPrev.shape[1]
                if (prevSeqLen < seqLen) {
                    const paddings = [
                        [0, 0],
                        [seqLen - prevSeqLen, 0],
                        [0, 0]
                    ]
                    const hPr = this.hPrev.clone()
                    this.hPrev.dispose()
                    this.hPrev = hPr.pad(paddings, 1)
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
            }

            let hInitial = this.ops.applyDense(
                inputs,
                this.C.read(),
                this.b.read()
            )
            hInitial = hInitial.add(
                this.ops.applyDense(this.hPrev, this.W.read())
            )

            hInitial = this.ops.rmsNorm(hInitial)

            hInitial = tf.layers
                .activation({ activation: this.activation })
                .apply(hInitial)

            let h = hInitial
            for (let s = 0; s < this.steps; s++) {
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

            while (this.hHistory.length > this.steps) {
                this.hHistory[0].dispose()
                this.hHistory.shift()
            }

            this.hPrev.dispose()
            this.hPrev = tf.keep(h.clone())
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
            steps: this.steps,
            learningRate: this.learningRate,
            decayRate: this.decayRate
        }
    }
}

tf.serialization.registerClass(FastAssociativeMemory)

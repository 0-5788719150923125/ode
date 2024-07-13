import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

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
            tf.initializers.glorotNormal()
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

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const seqLen = inputs.shape[1]

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
                    this.hPrev = this.hPrev.pad(paddings, 1)
                    this.hHistory = this.hHistory.map((h) =>
                        tf.keep(h.pad(paddings, 1))
                    )
                } else if (prevSeqLen > seqLen) {
                    const paddings = [
                        [0, 0],
                        [prevSeqLen - seqLen, 0],
                        [0, 0]
                    ]
                    inputs = inputs.pad(paddings, 0)
                }
            }

            let hInitial = this.applyDense(inputs, this.C, this.b)
            hInitial = hInitial.add(this.applyDense(this.hPrev, this.W))

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

            this.hPrev = tf.keep(h)
            this.hHistory.push(tf.keep(h))

            return inputs.add(h)
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

import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2005.00743
// https://github.com/iafarhan/causal-synthesizer-multihead-attention/blob/main/synthesizer.py
export default class SynthesizerAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
        this.heads = config.heads
        this.blockSize = config.blockSize
        this.attnPdrop = config.dropout || 0.0
        this.residPdrop = config.dropout || 0.0
        this.activation = tf.leakyRelu
        this.epsilon = config.epsilon || false
        this.alpha = config.alpha || 1
        this.depth = this.units / this.heads
    }

    build(inputShape) {
        this.w1 = this.addWeight(
            `w1`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.w2 = this.addWeight(
            `w2`,
            [this.units / this.heads, this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.b2 = this.addWeight(
            `b2`,
            [this.blockSize],
            'float32',
            tf.initializers.zeros()
        )
        this.value = this.addWeight(
            `value`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.proj = this.addWeight(
            `proj`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )

        if (this.epsilon) {
            this.layernorm = tf.layers.layerNormalization({
                epsilon: this.epsilon
            })
        }

        this.attnDropout = tf.layers.dropout({ rate: this.attnPdrop })
        this.residDropout = tf.layers.dropout({ rate: this.residPdrop })

        const mask = tf.linalg.bandPart(
            tf.ones([this.blockSize, this.blockSize]),
            -1,
            0
        )
        this.mask = tf.keep(tf.expandDims(tf.expandDims(mask, 0), 0))
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, seqLen, embedSize] = inputs.shape

            const nonlinearOut = this.activation(
                this.synthesize(inputs, this.w1.read()),
                this.alpha
            )
            const nonlinearReshaped = tf.transpose(
                tf.reshape(nonlinearOut, [
                    batchSize,
                    seqLen,
                    this.heads,
                    this.depth
                ]),
                [0, 2, 1, 3]
            )

            const v = this.synthesize(inputs, this.value.read())
            const vReshaped = tf.transpose(
                tf.reshape(v, [batchSize, seqLen, this.heads, this.depth]),
                [0, 2, 1, 3]
            )

            const w2Tiled = this.w2
                .read()
                .expandDims(0)
                .tile([batchSize * this.heads, 1, 1])
            let scores = tf.add(
                tf.reshape(
                    tf.matMul(
                        tf.reshape(nonlinearReshaped, [
                            batchSize * this.heads,
                            seqLen,
                            this.depth
                        ]),
                        w2Tiled
                    ),
                    [batchSize, this.heads, seqLen, this.blockSize]
                ),
                this.b2.read()
            )

            scores = scores.slice(
                [0, 0, 0, 0],
                [batchSize, this.heads, seqLen, seqLen]
            )

            scores = tf.where(
                tf.equal(
                    this.mask.slice([0, 0, 0, 0], [1, 1, seqLen, seqLen]),
                    0
                ),
                tf.fill([batchSize, this.heads, seqLen, seqLen], -1e10),
                scores
            )

            const probAttn = tf.softmax(scores, -1)
            const attnOutput = this.attnDropout.apply(probAttn)
            const y = tf.matMul(attnOutput, vReshaped)

            const yTransposed = tf.transpose(y, [0, 2, 1, 3])
            const yReshaped = tf.reshape(yTransposed, [
                batchSize,
                seqLen,
                embedSize
            ])

            let output = this.synthesize(yReshaped, this.proj.read())

            output = this.residDropout.apply(output)
            if (this.layernorm) output = this.layernorm.apply(output)

            return tf.add(inputs, output)
        })
    }

    synthesize(x, kernel) {
        const k = kernel.expandDims(0).tile([x.shape[0], 1, 1])
        return tf.matMul(x, k)
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.blockSize, this.units]
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            heads: this.heads,
            blockSize: this.blockSize,
            attnPdrop: this.dropout,
            residPdrop: this.dropout,
            epsilon: this.epsilon,
            alpha: this.alpha,
            depth: this.depth
        })
        return config
    }
}

tf.serialization.registerClass(SynthesizerAttention)

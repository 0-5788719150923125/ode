import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'
import { randomString } from '../utils.js'

export default class GPT2Attention extends LayerBase {
    constructor(config) {
        super(config)
        this.config = config

        // Config
        this.blockSize = config.blockSize || 256
        this.units = config.units || 256
        this.heads = config.heads || 4
        this.dropout = config.dropout || 0
        this.bias = config.bias || false
        this.epsilon = config.epsilon || 1e-5
        // Causal mask
        this.mask = tf.keep(
            tf.linalg.bandPart(
                tf.ones([config.blockSize, config.blockSize]),
                -1,
                0
            )
        )
    }

    build(inputShape) {
        this.cAttnKernel = this.addWeight(
            `c_attn-${randomString()}`,
            [this.units, 3 * this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        this.cAttnBias = this.addWeight(
            `c_attn-${randomString()}`,
            [3 * this.units],
            'float32',
            this.initializers.zeros()
        )
        this.cProjKernel = this.addWeight(
            `c_proj-${randomString()}`,
            [this.units, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        this.cProjBias = this.addWeight(
            `c_proj-${randomString()}`,
            [this.units],
            'float32',
            this.initializers.zeros()
        )
        this.layerNorm = tf.layers.layerNormalization({ epsilon: this.epsilon })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        // This is needed to save and load the model
        // When the model is saved, the config is saved with it
        // When the model is loaded, the config is used to create a new instance of the layer
        const config = super.getConfig()
        return Object.assign({}, config, this.config)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0]
            }

            const cAttn = this.ops.applyDense(
                inputs,
                this.cAttnKernel.read(),
                this.cAttnBias.read()
            )

            // Make order of qkv split to follow minGPT
            let [q, k, v] = tf.split(cAttn, 3, -1)
            const [B, T, C] = k.shape

            const splitHeads = (x) =>
                tf.transpose(
                    tf.reshape(x, [B, T, this.heads, C / this.heads]),
                    [0, 2, 1, 3]
                )

            q = splitHeads(q)
            k = splitHeads(k)
            v = splitHeads(v)

            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            let att = tf.mul(
                tf.matMul(q, k, false, true),
                tf.div(
                    1,
                    tf.sqrt(tf.cast(k.shape[k.shape.length - 1], 'float32'))
                )
            )

            const mask = this.mask.slice([0, 0], [T, T])
            att = tf.add(att, tf.mul(tf.sub(1, mask), -1e9))
            att = tf.softmax(att, -1)
            att = kwargs['training'] ? tf.dropout(att, this.dropout) : att

            let outputs = tf.matMul(att, v)

            outputs = tf.transpose(outputs, [0, 2, 1, 3])
            outputs = tf.reshape(outputs, [B, T, C])
            outputs = this.ops.applyDense(
                outputs,
                this.cProjKernel.read(),
                this.cProjBias.read()
            )
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            outputs = this.layerNorm.apply(outputs)
            return tf.layers.add().apply([inputs, outputs])
        })
    }
}

tf.serialization.registerClass(GPT2Attention)

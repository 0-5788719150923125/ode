import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class PEER extends LayerBase {
    constructor(config) {
        super(config)
        this.numExperts = config.numExperts || 1024
        this.hiddenDim = config.hiddenDim || 1
        this.numHeads = config.numHeads || 8
        this.topK = config.topK || 16
        this.dropout = config.dropout || 0
        this.activation = config.activation || 'relu'
        this.sparsityRatio = config.sparsityRatio || 0.9
    }
    build(inputShape) {
        const numSubKeys = Math.floor(Math.sqrt(this.numExperts))
        this.units = inputShape[inputShape.length - 1]
        this.subKeys = []
        this.subKeys.push(
            this.addWeight(
                `subKeys1`,
                [numSubKeys, this.units / 2],
                'float32',
                tf.initializers.glorotNormal({ seed: this.ops.getSeed() })
            )
        )
        this.subKeys.push(
            this.addWeight(
                `subKeys2`,
                [numSubKeys, this.units / 2],
                'float32',
                tf.initializers.glorotNormal({ seed: this.ops.getSeed() })
            )
        )
        this.downProj = this.addWeight(
            `downProj`,
            [this.numExperts, this.units],
            'float32',
            tf.initializers.glorotNormal({ seed: this.ops.getSeed() })
        )
        this.upProj = this.addWeight(
            `upProj`,
            [this.numExperts, this.units],
            'float32',
            tf.initializers.glorotNormal({ seed: this.ops.getSeed() })
        )
        this.queryKernel = this.addWeight(
            `queryKernel`,
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal({ seed: this.ops.getSeed() })
        )
        if (this.useBias) {
            this.queryBias = this.addWeight(
                `queryBias`,
                [this.units],
                'float32',
                tf.initializers.zeros()
            )
        }
    }

    sparseTopKWithSTE(inputs, sparsityRatio) {
        return tf.customGrad((inputs, save) => {
            const k = Math.floor(
                inputs.shape[inputs.shape.length - 1] * sparsityRatio
            )
            const topK = tf.topk(tf.abs(inputs), k)
            const mask = tf.greaterEqual(tf.abs(inputs), topK.values.min())
            const sparseOutputs = tf.mul(inputs, tf.cast(mask, inputs.dtype))
            save([inputs])
            return {
                value: sparseOutputs,
                gradFunc: (dy, saved) => [dy]
            }
        })(inputs)
    }

    computeProductKeysMask(query) {
        const queryHeads = tf.split(query, this.numHeads, -1)
        const subQueries = queryHeads.map((q) => {
            const subQuery = tf.split(q, 2, -1)
            const subQueryMask = []
            for (let i = 0; i < 2; i++) {
                const scores = tf.matMul(
                    subQuery[i],
                    this.subKeys[i].read(),
                    false,
                    true
                )
                const topKIndices = tf.topk(scores, this.topK).indices
                const subMask = tf.oneHot(topKIndices, this.subKeys[i].shape[0])
                subQueryMask.push(subMask)
            }
            const headMask = tf.reshape(
                tf.einsum('ik,jk->ijk', subQueryMask[0], subQueryMask[1]),
                [-1, this.topK * this.topK]
            )
            return headMask
        })
        const productKeysMask = tf.concat(subQueries, -1)
        return productKeysMask
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const query = this.ops.applyDense(
                inputs,
                this.queryKernel.read(),
                this.queryBias?.read()
            )
            const productKeysMask = this.computeProductKeysMask(query)
            const expertIndices = tf.reshape(
                tf.cast(tf.argMax(productKeysMask, -1), 'int32'),
                [-1, 1]
            )
            const routerScores = tf.gather(
                tf.softmax(productKeysMask),
                expertIndices,
                1
            )
            const downProjWeights = tf.gathering.gather(
                this.downProj.read(),
                expertIndices
            )
            const upProjWeights = tf.gathering.gather(
                this.upProj.read(),
                expertIndices
            )

            const hiddenStates = this.ops.applyDense(inputs, downProjWeights)
            const activations = tf.layers
                .activation({
                    activation: this.activation
                })
                .apply(hiddenStates)
            const expertsOutputs = this.sparseTopKWithSTE(
                this.ops.applyDense(activations, upProjWeights),
                this.sparsityRatio
            )
            const outputs = tf.sum(tf.mul(expertsOutputs, routerScores), -2)

            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs

            return outputs
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            numHeads: this.numHeads,
            topK: this.topK,
            dropout: this.dropout,
            activation: this.activation,
            sparsityRatio: this.sparsityRatio
        }
    }
}
tf.serialization.registerClass(PEER)

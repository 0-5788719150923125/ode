import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
// currently failing, because the causal mask makes loss values extremely high
export default class EfficientAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.keyChannels = config.keyChannels || 256
        this.valueChannels = config.valueChannels || 256
        this.headCount = config.headCount || 8
        this.contextLength = config.contextLength
    }

    build(inputShape) {
        const inputDepth = inputShape[inputShape.length - 1]

        this.queries = this.addWeight(
            'queries',
            [1, inputDepth, this.keyChannels],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
        this.keys = this.addWeight(
            'keys',
            [1, inputDepth, this.keyChannels],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
        this.values = this.addWeight(
            'values',
            [1, inputDepth, this.valueChannels],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
        this.reprojection = this.addWeight(
            'reprojection',
            [1, this.valueChannels, inputDepth],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ops.getSeed()
            })
        )
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, dims] = inputs.shape

            const queries = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.queries.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.keyChannels, seqLen])
            const keys = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.keys.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.keyChannels, seqLen])
            const values = tf
                .conv1d(
                    inputs.reshape([batchSize, seqLen, dims]),
                    this.values.read(),
                    1,
                    'same'
                )
                .reshape([batchSize, this.valueChannels, seqLen])

            const headKeyChannels = Math.floor(
                this.keyChannels / this.headCount
            )
            const headValueChannels = Math.floor(
                this.valueChannels / this.headCount
            )

            const mask = tf.linalg
                .bandPart(
                    tf.ones([this.contextLength, this.contextLength]),
                    0,
                    -1
                )
                .sub(tf.eye(this.contextLength))
                .mul(tf.scalar(-1e9))

            const attendedValues = []
            for (let i = 0; i < this.headCount; i++) {
                const key = keys
                    .slice(
                        [0, i * headKeyChannels, 0],
                        [batchSize, headKeyChannels, seqLen]
                    )
                    .softmax(-1)
                const query = queries
                    .slice(
                        [0, i * headKeyChannels, 0],
                        [batchSize, headKeyChannels, seqLen]
                    )
                    .transpose([0, 2, 1])
                    .softmax(-1)
                    .transpose([0, 2, 1])
                const value = values.slice(
                    [0, i * headValueChannels, 0],
                    [batchSize, headValueChannels, seqLen]
                )

                const context = tf.matMul(key, value, false, true).add(mask)
                const attendedValue = tf
                    .matMul(context, query, true, false)
                    .reshape([batchSize, headValueChannels, seqLen])

                attendedValues.push(attendedValue)
            }

            const aggregatedValues = tf
                .concat(attendedValues, 1)
                .reshape([batchSize, seqLen, this.valueChannels])

            const outputs = tf.conv1d(
                aggregatedValues,
                this.reprojection.read(),
                1,
                'same'
            )

            return tf.add(inputs, outputs)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            keyChannels: this.keyChannels,
            valueChannels: this.valueChannels,
            headCount: this.headCount
        }
    }
}

tf.serialization.registerClass(EfficientAttention)

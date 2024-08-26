import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class LocalSensitiveHashingAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
        this.projectionDim = config.projectionDim || 32
        this.numBuckets = config.numBuckets || 64
        this.numHashes = config.numHashes || 4
        this.outputDim = config.outputDim || null
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const outputDim = this.outputDim || inputDim
        this.outputDim = outputDim

        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputDim, this.numHeads * this.headDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputDim, this.numHeads * this.headDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputDim, this.numHeads * this.headDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.projectionKernel = this.addWeight(
            'projectionKernel',
            [this.numHeads, this.headDim, this.projectionDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.numHeads * this.headDim, outputDim],
            'float32',
            tf.initializers.glorotUniform({
                seed: this.ode.ops.getSeed()
            })
        )
        if (this.useBias) {
            this.queryBias = this.addWeight(
                'queryBias',
                [this.numHeads * this.headDim],
                'float32',
                tf.initializers.zeros()
            )
            this.keyBias = this.addWeight(
                'keyBias',
                [this.numHeads * this.headDim],
                'float32',
                tf.initializers.zeros()
            )
            this.valueBias = this.addWeight(
                'valueBias',
                [this.numHeads * this.headDim],
                'float32',
                tf.initializers.zeros()
            )
            this.projectionBias = this.addWeight(
                'projectionBias',
                [this.numHeads, this.projectionDim],
                'float32',
                tf.initializers.zeros()
            )
            this.outputBias = this.addWeight(
                'outputBias',
                [outputDim],
                'float32',
                tf.initializers.zeros()
            )
        }

        this.lshHashKernels = []
        for (let i = 0; i < this.numHashes; i++) {
            this.lshHashKernels.push(
                this.addWeight(
                    `lshHashKernel${i}`,
                    [this.projectionDim, this.numBuckets],
                    'float32',
                    tf.initializers.glorotUniform({
                        seed: this.ode.ops.getSeed()
                    })
                )
            )
        }
    }

    applyLSH(projectedKeys, projectedQueries) {
        return tf.customGrad((projectedKeys, projectedQueries, save) => {
            const hashedKeys = []
            const hashedQueries = []
            for (let i = 0; i < this.numHashes; i++) {
                const kernel = this.lshHashKernels[i].read()
                const keyScores = tf.matMul(projectedKeys, kernel)
                const queryScores = tf.matMul(projectedQueries, kernel)
                const keyBuckets = tf.argMax(keyScores, -1)
                const queryBuckets = tf.argMax(queryScores, -1)
                hashedKeys.push(keyBuckets)
                hashedQueries.push(queryBuckets)
            }
            const hashedKeysStacked = tf.stack(hashedKeys, -1)
            const hashedQueriesStacked = tf.stack(hashedQueries, -1)

            const concatenatedTensor = tf.concat(
                [hashedKeysStacked, hashedQueriesStacked],
                -1
            )

            save([projectedKeys, projectedQueries])

            return {
                value: concatenatedTensor,
                gradFunc: (dy, saved) => {
                    const [savedKeys, savedQueries] = saved
                    const zeroGrad = tf.zerosLike(concatenatedTensor)
                    return [tf.zerosLike(savedKeys), tf.zerosLike(savedQueries)]
                }
            }
        })(projectedKeys, projectedQueries)
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, inputDim] = inputs.shape

            const Q = this.ops.applyDense(
                inputs,
                this.queryKernel.read(),
                this.queryBias?.read()
            )
            const K = this.ops.applyDense(
                inputs,
                this.keyKernel.read(),
                this.keyBias?.read()
            )
            const V = this.ops.applyDense(
                inputs,
                this.valueKernel.read(),
                this.valueBias?.read()
            )

            const QHeads = tf.reshape(Q, [
                batchSize,
                seqLen,
                this.numHeads,
                this.headDim
            ])
            const KHeads = tf.reshape(K, [
                batchSize,
                seqLen,
                this.numHeads,
                this.headDim
            ])
            const VHeads = tf.reshape(V, [
                batchSize,
                seqLen,
                this.numHeads,
                this.headDim
            ])

            const QHeadsTransposed = tf.transpose(QHeads, [0, 2, 1, 3])
            const KHeadsTransposed = tf.transpose(KHeads, [0, 2, 1, 3])
            const VHeadsTransposed = tf.transpose(VHeads, [0, 2, 1, 3])

            const tileShape = [batchSize, 1, 1, 1]
            const projectionKernel = this.projectionKernel
                .read()
                .expandDims(0)
                .tile(tileShape)
            const projectionBias = this.projectionBias
                ?.read()
                .reshape([1, this.numHeads, 1, this.projectionDim])

            const KP = tf.matMul(KHeadsTransposed, projectionKernel)
            const QP = tf.matMul(QHeadsTransposed, projectionKernel)
            if (this.useBias) {
                KP.add(projectionBias)
                QP.add(projectionBias)
            }

            const concatenatedTensor = this.applyLSH(KP, QP)
            const KPHashed = concatenatedTensor.slice(
                [0, 0, 0, 0],
                [batchSize, this.numHeads, seqLen, this.numHashes]
            )
            const QPHashed = concatenatedTensor.slice(
                [0, 0, 0, this.numHashes],
                [batchSize, this.numHeads, seqLen, this.numHashes]
            )

            const scores = tf
                .cast(
                    tf.pow(
                        tf.sub(QPHashed.expandDims(3), KPHashed.expandDims(2)),
                        2
                    ),
                    'float32'
                )
                .mul(tf.scalar(-1))
                .sum(-1)

            const normalizedScores = tf.div(
                scores,
                tf.sqrt(tf.scalar(this.numHashes))
            )

            const mask = tf.linalg
                .bandPart(tf.ones([seqLen, seqLen]), 0, -1)
                .sub(tf.eye(seqLen))
                .mul(tf.scalar(-1e9))

            const maskedScores = tf.add(
                normalizedScores,
                mask.expandDims(0).expandDims(0)
            )

            const weights = tf.softmax(maskedScores, -1)

            const weightedValues = tf.matMul(weights, VHeadsTransposed)

            const outputHeads = tf.transpose(weightedValues, [0, 2, 1, 3])
            const output = tf.reshape(outputHeads, [
                batchSize,
                seqLen,
                this.numHeads * this.headDim
            ])

            const finalOutput = this.ops.applyDense(
                output,
                this.outputKernel.read(),
                this.outputBias?.read()
            )

            return tf.add(inputs, this.ops.rmsNorm(finalOutput))
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numHeads: this.numHeads,
            headDim: this.headDim,
            projectionDim: this.projectionDim,
            numBuckets: this.numBuckets,
            numHashes: this.numHashes,
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(LocalSensitiveHashingAttention)

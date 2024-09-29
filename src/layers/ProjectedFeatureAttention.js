import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class ProjectedFeatureAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.numHeads = config.numHeads || 8
        this.headDim = config.headDim || 64
        this.headFeatures = config.headFeatures || 32
        this.queriesPerHead = config.queriesPerHead || 1
        this.outputDim = config.outputDim || null
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const outputDim = this.outputDim || inputDim
        this.outputDim = outputDim

        if (outputDim > inputDim) {
            this.inProjKernel = this.addWeight(
                'inProjKernel',
                [inputDim, outputDim],
                'float32',
                this.initializers.glorotUniform()
            )
            if (this.useBias)
                this.inProjBias = this.addWeight(
                    `inProjBias`,
                    [outputDim],
                    'float32',
                    this.initializers.zeros()
                )
        }

        this.queryKernel = this.addWeight(
            'queryKernel',
            [inputDim, this.numHeads * this.queriesPerHead * this.headFeatures],
            'float32',
            this.initializers.glorotUniform()
        )
        this.keyKernel = this.addWeight(
            'keyKernel',
            [inputDim, this.numHeads * this.headDim],
            'float32',
            this.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            'valueKernel',
            [inputDim, this.numHeads * this.headDim],
            'float32',
            this.initializers.glorotUniform()
        )
        this.projectionKernel = this.addWeight(
            'projectionKernel',
            [this.numHeads, this.headDim, this.headFeatures],
            'float32',
            this.initializers.glorotUniform()
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [
                this.numHeads * this.queriesPerHead * this.headFeatures,
                outputDim
            ],
            'float32',
            this.initializers.glorotUniform()
        )
        if (this.useBias) {
            this.queryBias = this.addWeight(
                'queryBias',
                [this.numHeads * this.queriesPerHead * this.headFeatures],
                'float32',
                this.initializers.zeros()
            )
            this.keyBias = this.addWeight(
                'keyBias',
                [this.numHeads * this.headDim],
                'float32',
                this.initializers.zeros()
            )
            this.valueBias = this.addWeight(
                'valueBias',
                [this.numHeads * this.headDim],
                'float32',
                this.initializers.zeros()
            )
            this.projectionBias = this.addWeight(
                'projectionBias',
                [this.numHeads, 1, this.headFeatures],
                'float32',
                this.initializers.zeros()
            )
            this.outputBias = this.addWeight(
                'outputBias',
                [outputDim],
                'float32',
                this.initializers.zeros()
            )
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const [batchSize, seqLen, inputDim] = inputs.shape

            let projectedInputs = inputs
            if (this.inProjKernel) {
                projectedInputs = this.ops.applyDense(
                    inputs,
                    this.inProjKernel.read(),
                    this.inProjBias?.read()
                )
            }

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
                this.queriesPerHead,
                this.headFeatures
            ])
            const kvShape = [batchSize, seqLen, this.numHeads, this.headDim]
            const KHeads = tf.reshape(K, kvShape)
            const VHeads = tf.reshape(V, kvShape)

            const QHeadsTransposed = tf.transpose(QHeads, [0, 2, 3, 1, 4])
            const KHeadsTransposed = tf.transpose(KHeads, [0, 2, 1, 3])
            const VHeadsTransposed = tf.transpose(VHeads, [0, 2, 1, 3])

            const tileShape = [batchSize, 1, 1, 1]
            const projectionKernel = this.projectionKernel
                .read()
                .expandDims(0)
                .tile(tileShape)

            let KP = tf.matMul(KHeadsTransposed, projectionKernel)
            let VP = tf.matMul(VHeadsTransposed, projectionKernel)

            if (this.useBias) {
                const biasShape = [1, this.numHeads, 1, this.headFeatures]
                const projectionBias = this.projectionBias
                    ?.read()
                    .reshape(biasShape)
                KP = tf.add(KP, projectionBias)
                VP = tf.add(VP, projectionBias)
            }

            // Reshape KP and VP to [batchSize, numHeads, seqLen, headFeatures]
            const headShape = [
                batchSize,
                this.numHeads,
                seqLen,
                this.headFeatures
            ]
            const KPReshaped = tf.reshape(KP, headShape)
            const VPReshaped = tf.reshape(VP, headShape)

            // Tile KP and VP to match the number of queries per head
            const KPTiled = tf.tile(KPReshaped, [1, 1, this.queriesPerHead, 1])
            const VPTiled = tf.tile(VPReshaped, [1, 1, this.queriesPerHead, 1])

            // Reshape tiled KP and VP to match QHeadsReshaped dimensions
            const projectionShape = [
                batchSize,
                this.numHeads * this.queriesPerHead,
                seqLen,
                this.headFeatures
            ]
            const KPTiledReshaped = tf.reshape(KPTiled, projectionShape)
            const VPTiledReshaped = tf.reshape(VPTiled, projectionShape)
            const QHeadsReshaped = tf.reshape(QHeadsTransposed, projectionShape)

            const scores = tf.matMul(
                QHeadsReshaped,
                KPTiledReshaped,
                false,
                true
            )
            let normalizedScores = tf.mul(
                scores,
                tf.rsqrt(tf.scalar(this.headFeatures))
            )

            // Apply ALiBi if needed
            if (this.ALiBiLength) {
                // Reshape normalizedScores to 5D tensor
                const normalizedScoresReshaped = tf.reshape(normalizedScores, [
                    batchSize,
                    this.numHeads,
                    this.queriesPerHead,
                    seqLen,
                    seqLen
                ])

                // Apply ALiBi to the reshaped tensor
                const scoresWithALiBi = this.ops.applyALiBi(
                    normalizedScoresReshaped,
                    this.numHeads,
                    this.queriesPerHead,
                    this.ALiBiLength
                )

                // Reshape the scores back to 4D tensor
                normalizedScores = tf.reshape(scoresWithALiBi, [
                    batchSize,
                    this.numHeads * this.queriesPerHead,
                    seqLen,
                    seqLen
                ])
            }

            const mask = tf.linalg
                .bandPart(tf.ones([seqLen, seqLen]), 0, -1)
                .sub(tf.eye(seqLen))
                .mul(tf.scalar(-1e9))
            const maskedScores = tf.add(normalizedScores, mask)

            const weights = tf.softmax(maskedScores, -1)

            const weightedValues = tf.matMul(weights, VPTiledReshaped)

            const outputHeads = tf.reshape(weightedValues, [
                batchSize,
                seqLen,
                this.numHeads * this.queriesPerHead * this.headFeatures
            ])

            const outputs = this.ops.applyDense(
                outputHeads,
                this.outputKernel.read(),
                this.outputBias?.read()
            )

            return outputs
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
            headFeatures: this.headFeatures,
            queriesPerHead: this.queriesPerHead,
            outputDim: this.outputDim
        }
    }
}

tf.serialization.registerClass(ProjectedFeatureAttention)

import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class LowRankFactorization extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
        this.rank = config.rank || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.leftMatrix = this.addWeight(
            'leftMatrix',
            [inputDim, this.rank],
            'float32',
            this.initializers.glorotNormal()
        )

        this.rightMatrix = this.addWeight(
            'rightMatrix',
            [this.rank, this.units],
            'float32',
            this.initializers.glorotNormal()
        )

        this.residualMatrix = this.addWeight(
            'residualMatrix',
            [inputDim, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
    }

    call(inputs) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        const [batchSize, seqLength, inputDim] = inputs.shape

        const inputReshaped = inputs.reshape([-1, inputDim])
        const lowRankOutput = inputReshaped
            .matMul(this.leftMatrix.read())
            .matMul(this.rightMatrix.read())

        const residualOutput = inputReshaped.matMul(this.residualMatrix.read())

        const output = lowRankOutput.add(residualOutput)

        return output.reshape([batchSize, seqLength, this.units])
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            rank: this.rank
        }
    }
}

tf.serialization.registerClass(LowRankFactorization)

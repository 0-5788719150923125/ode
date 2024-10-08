import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class AttentionBasedReduction extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config.units
        this.hiddenDim = config.hiddenDim || 64
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryWeights = this.addWeight(
            'queryWeights',
            [inputDim, this.hiddenDim],
            'float32',
            this.initializers.glorotNormal()
        )

        this.keyWeights = this.addWeight(
            'keyWeights',
            [inputDim, this.hiddenDim],
            'float32',
            this.initializers.glorotNormal()
        )

        this.valueWeights = this.addWeight(
            'valueWeights',
            [inputDim, this.units],
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

        const queryMatrix = inputReshaped.matMul(this.queryWeights.read())
        const keyMatrix = inputReshaped.matMul(this.keyWeights.read())
        const valueMatrix = inputReshaped.matMul(this.valueWeights.read())

        const attentionScores = queryMatrix.matMul(keyMatrix.transpose())
        const attentionWeights = attentionScores.softmax()

        const attentionOutput = attentionWeights.matMul(valueMatrix)

        const residualOutput = inputReshaped.matMul(this.residualMatrix.read())

        return attentionOutput
            .add(residualOutput)
            .reshape([batchSize, seqLength, this.units])
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.units]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            hiddenDim: this.hiddenDim
        }
    }
}

tf.serialization.registerClass(AttentionBasedReduction)

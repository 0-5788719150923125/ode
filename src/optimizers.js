import * as tf from '@tensorflow/tfjs'

class AdamW extends tf.AdamOptimizer {
    constructor(
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        decayRate = 1e-4,
        includeInWeightDecay,
        excludeFromWeightDecay
    ) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.decayRate = decayRate
        this.includeInWeightDecay = includeInWeightDecay
        this.excludeFromWeightDecay = excludeFromWeightDecay
    }
    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Array.isArray(variableGradients)
                ? variableGradients.map((v) => v.name)
                : Object.keys(variableGradients)

            // Apply weight decay
            varNames.forEach((name, i) => {
                if (this.includeInWeightDecay.includes(name)) {
                    const value = this.ENGINE.registeredVariables[name]
                    const newValue = tf.sub(
                        value,
                        tf.mul(this.learningRate, tf.mul(value, this.decayRate))
                    )
                    value.assign(newValue)
                }
            })

            super.applyGradients(variableGradients)
        })
    }
}

function prepareAdamW(model, learningRate, beta1, beta2, epsilon, decayRate) {
    const includeInWeightDecay = []
    const excludeFromWeightDecay = []

    if (decayRate <= 0) {
        throw 'AdamW with a decayRate of 0 is just regular Adam. You should use `tf.train.adam` instead.'
    } else {
        model.getNamedWeights().forEach((v) => {
            const name = v.name.toLowerCase()
            if (
                name.includes('bias') ||
                name.includes('norm') ||
                name.includes('emb')
            ) {
                excludeFromWeightDecay.push(v.name)
            } else {
                includeInWeightDecay.push(v.name)
            }
        })
        return new AdamW(
            learningRate,
            beta1,
            beta2,
            epsilon,
            decayRate,
            includeInWeightDecay,
            excludeFromWeightDecay
        )
    }
}

const customOptimizers = {
    AdamW: (model, learningRate, beta1, beta2, epsilon, decayRate) =>
        prepareAdamW(model, learningRate, beta1, beta2, epsilon, decayRate)
}

export default customOptimizers

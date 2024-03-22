import tf from '@tensorflow/tfjs'

const ENGINE = tf.engine()

export function getAdamW(
    model,
    learningRate,
    beta1,
    beta2,
    epsilon,
    decayRate
) {
    const includeInWeightDecay = []
    const excludeFromWeightDecay = []

    if (decayRate <= 0) {
        throw 'AdamW with a decayRate of 0 is just Adam. You should use the `tf.train.adam` optimizer instead.'
    } else {
        model.getNamedWeights().forEach((v) => {
            if (
                v.name.toLowerCase().includes('bias') ||
                v.name.toLowerCase().includes('norm') ||
                v.name.toLowerCase().includes('emb')
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
                    const value = ENGINE.registeredVariables[name]
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

import * as tf from '@tensorflow/tfjs'

// AdamG: a parameter-free optimizer
// https://arxiv.org/abs/2405.04376
export default class AdamG extends tf.Optimizer {
    constructor({
        learningRate = 1.0,
        beta1 = 0.95,
        beta2 = 0.999,
        beta3 = 0.95,
        accBeta1 = 1,
        accBeta2 = 1,
        epsilon = 1e-8,
        p = 0.2,
        q = 0.24,
        step = 0
    } = {}) {
        super()
        this.learningRate = learningRate
        this.beta1 = beta1
        this.beta2 = beta2
        this.beta3 = beta3
        this.epsilon = epsilon
        this.p = p
        this.q = q
        this.accBeta1 = accBeta1
        this.accBeta2 = accBeta2
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const learningRateScaled = Math.min(
                this.learningRate,
                1 / Math.sqrt(this.step + 1)
            )

            const variableNames = Array.isArray(variableGradients)
                ? variableGradients.map((v) => v.name)
                : Object.keys(variableGradients)

            variableNames.forEach((name) => {
                const value = this.ENGINE.registeredVariables[name]
                const grad = variableGradients[name]

                if (!this.STATE[name]) {
                    this.STATE[name] = {
                        firstMoment: tf.variable(tf.zerosLike(value)),
                        secondMoment: tf.variable(tf.zerosLike(value)),
                        goldenStep: tf.variable(tf.zerosLike(value))
                    }
                }

                const { firstMoment, secondMoment, goldenStep } =
                    this.STATE[name]

                const newGoldenStep = goldenStep.mul(this.beta3).add(
                    tf
                        .scalar(this.p)
                        .mul(secondMoment.pow(this.q))
                        .mul(1 - this.beta3)
                )

                const newFirstMoment = firstMoment
                    .mul(this.beta1)
                    .add(grad.mul(newGoldenStep).mul(1 - this.beta1))
                const newSecondMoment = secondMoment
                    .mul(this.beta2)
                    .add(grad.square().mul(1 - this.beta2))

                const biasCorrectedFirstMoment = newFirstMoment.div(
                    tf.scalar(1).sub(this.accBeta1).add(this.epsilon)
                )
                const biasCorrectedSecondMoment = newSecondMoment.div(
                    tf.scalar(1).sub(this.accBeta2).add(this.epsilon)
                )

                const update = biasCorrectedFirstMoment
                    .div(biasCorrectedSecondMoment.sqrt().add(this.epsilon))
                    .mul(tf.scalar(learningRateScaled))

                value.assign(value.sub(update))

                this.STATE[name].firstMoment.assign(newFirstMoment)
                this.STATE[name].secondMoment.assign(newSecondMoment)
                this.STATE[name].goldenStep.assign(newGoldenStep)
            })
        })

        this.accBeta1 *= this.beta1
        this.accBeta2 *= this.beta2
        this.step++
        this.incrementIterations()
    }

    setWeights(weightValues) {
        weightValues.forEach((namedTensor) => {
            const [name, tensorName] = namedTensor.name.split('__')
            if (!this.STATE[name]) this.STATE[name] = {}
            this.STATE[name][tensorName] = tf.variable(namedTensor.tensor)
        })
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).forEach(([name, state]) => {
            weights.push({
                name: `${name}__firstMoment`,
                tensor: state.firstMoment
            })
            weights.push({
                name: `${name}__secondMoment`,
                tensor: state.secondMoment
            })
            weights.push({
                name: `${name}__goldenStep`,
                tensor: state.goldenStep
            })
        })
        return weights
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            beta3: this.beta3,
            accBeta1: this.accBeta1,
            accBeta2: this.accBeta2,
            epsilon: this.epsilon,
            p: this.p,
            q: this.q,
            step: this.step
        }
    }

    static get className() {
        return 'AdamG'
    }
}

tf.serialization.registerClass(AdamG)

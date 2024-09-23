import * as tf from '@tensorflow/tfjs'
import { applyWeightDecay } from './_ops.js'

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
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        step = 1
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
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const learningRateScaled = Math.min(
                this.learningRate,
                1 / Math.sqrt(this.step)
            )

            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]

                if (!this.STATE[name]) {
                    this.STATE[name] = {
                        firstMoment: tf.variable(tf.zerosLike(variable)),
                        secondMoment: tf.variable(tf.zerosLike(variable)),
                        goldenStep: tf.variable(tf.zerosLike(variable))
                    }
                }

                gradient = applyWeightDecay(
                    variable,
                    gradient,
                    name,
                    this.learningRate,
                    this.weightDecay,
                    this.weightDecouple,
                    this.fixedDecay
                )

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
                    .add(gradient.mul(newGoldenStep).mul(1 - this.beta1))
                    .div(tf.scalar(1).sub(this.accBeta1).add(this.epsilon))
                const newSecondMoment = secondMoment
                    .mul(this.beta2)
                    .add(gradient.square().mul(1 - this.beta2))
                    .div(tf.scalar(1).sub(this.accBeta2).add(this.epsilon))

                const update = newFirstMoment
                    .div(newSecondMoment.sqrt().add(this.epsilon))
                    .mul(tf.scalar(learningRateScaled))

                variable.assign(variable.sub(update))

                this.STATE[name].firstMoment.assign(newFirstMoment)
                this.STATE[name].secondMoment.assign(newSecondMoment)
                this.STATE[name].goldenStep.assign(newGoldenStep)
            })
        })

        this.accBeta1 *= this.beta1
        this.accBeta2 *= this.beta2

        this.incrementIterations()
        this.step++
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
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            step: this.step
        }
    }

    static get className() {
        return 'AdamG'
    }
}

tf.serialization.registerClass(AdamG)

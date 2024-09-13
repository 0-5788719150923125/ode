import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './_ops.js'

export default class SophiaH extends tf.Optimizer {
    constructor({
        learningRate = 6e-2,
        beta1 = 0.96,
        beta2 = 0.99,
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        gamma = 1e-2,
        updatePeriod = 10,
        numSamples = 1,
        hessianDistribution = 'gaussian',
        epsilon = 1e-12,
        step = 1
    } = {}) {
        super()
        this.learningRate = learningRate
        this.beta1 = beta1
        this.beta2 = beta2
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.gamma = gamma
        this.updatePeriod = updatePeriod
        this.numSamples = numSamples
        this.hessianDistribution = hessianDistribution
        this.epsilon = epsilon
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Object.keys(variableGradients)

            varNames.forEach((name) => {
                const variable = this.ENGINE.registeredVariables[name]
                const gradient = variableGradients[name]
                const state = this.STATE[name] || {
                    momentum: tf.variable(tf.zerosLike(variable)),
                    hessianMoment: tf.variable(tf.zerosLike(variable))
                }

                if (
                    this.weightDecay !== 0 &&
                    !shouldExcludeFromWeightDecay(name)
                ) {
                    if (this.weightDecouple) {
                        variable.assign(
                            variable.sub(
                                variable.mul(
                                    this.weightDecay * this.learningRate
                                )
                            )
                        )
                    } else if (!this.fixedDecay) {
                        gradient.assign(
                            gradient.add(variable.mul(this.weightDecay))
                        )
                    }
                }

                const momentum = state.momentum
                    .mul(this.beta1)
                    .add(gradient.mul(1 - this.beta1))

                if (this.step % this.updatePeriod === 0) {
                    const hessianEstimate = this.computeHutchinsonEstimator(
                        variable,
                        gradient
                    )
                    const hessianMoment = state.hessianMoment
                        .mul(this.beta2)
                        .add(hessianEstimate.mul(1 - this.beta2))
                    state.hessianMoment.assign(hessianMoment)
                }

                const effectiveHessian = tf.maximum(
                    state.hessianMoment,
                    this.epsilon
                )
                const update = momentum
                    .div(effectiveHessian)
                    .clipByValue(-this.gamma, this.gamma)
                variable.assign(variable.sub(update.mul(this.learningRate)))

                state.momentum.assign(momentum)
                this.STATE[name] = state
            })

            this.step++
        })
        this.incrementIterations()
    }

    computeHutchinsonEstimator(variable, gradient) {
        let hessianEstimate = tf.zerosLike(variable)

        for (let i = 0; i < this.numSamples; i++) {
            const u =
                this.hessianDistribution === 'gaussian'
                    ? tf.randomNormal(variable.shape)
                    : tf.randomUniform(variable.shape, -1, 1).sign()

            // Compute Hessian-vector product
            const hvp = tf
                .grad((v) => tf.sum(gradient.mul(v)))(variable)
                .mul(u)

            hessianEstimate = hessianEstimate.add(u.mul(hvp))
        }

        return hessianEstimate.div(this.numSamples)
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).map(([name, state]) => [
            {
                name: `${name}__momentum`,
                tensor: state.momentum.read()
            },
            {
                name: `${name}__hessianMoment`,
                tensor: state.hessianMoment.read()
            }
        ])
        return weights
    }

    setWeights(weightValues) {
        weightValues.forEach((namedTensor) => {
            const [name, tensorName] = namedTensor.name.split('__')
            if (!this.STATE[name]) this.STATE[name] = {}
            this.STATE[name][tensorName] = tf.variable(namedTensor.tensor)
        })
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            gamma: this.gamma,
            updatePeriod: this.updatePeriod,
            numSamples: this.numSamples,
            hessianDistribution: this.hessianDistribution,
            epsilon: this.epsilon,
            step: this.step
        }
    }

    static get className() {
        return 'SophiaH'
    }
}

tf.serialization.registerClass(SophiaH)

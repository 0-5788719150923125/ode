import * as tf from '@tensorflow/tfjs'
import { applyWeightDecay } from './_ops.js'
import { seededPRNG } from '../utils.js'

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
        epsilon = 1e-12,
        seed = null,
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
        this.epsilon = epsilon
        this.seed = seed
        this.useSeed = false
        if (this.seed !== null) {
            this.useSeed = true
        }
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]

                const state = this.STATE[name] || {
                    momentum: tf.variable(tf.zerosLike(variable)),
                    hessianMoment: tf.variable(tf.zerosLike(variable))
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

        if (this.useSeed) this.seed++

        for (let i = 0; i < this.numSamples; i++) {
            const mean = 0
            // const center = 0.23
            const stddv = 1
            let u
            if (i % 2 === 0) {
                // Gaussian distribution
                u = tf.randomNormal(
                    variable.shape,
                    mean,
                    stddv,
                    'float32',
                    seededPRNG(this.seed) ? this.seed : undefined
                )
            } else {
                // Rademacher distribution
                u = tf
                    .randomUniform(
                        variable.shape,
                        mean,
                        // center,
                        stddv,
                        'float32',
                        seededPRNG(this.seed + 1) ? this.seed : undefined // foresight
                    )
                    .round()
                    .mul(2)
                    .sub(1)
            }

            // Compute Hessian-vector product
            const hvp = tf.grad((v) => {
                const gradientDotU = gradient.mul(u).sum()
                return v.mul(gradientDotU)
            })(variable, u)

            hessianEstimate = hessianEstimate.add(u.mul(hvp))
        }

        return hessianEstimate.div(this.numSamples)
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).map(([name, state]) => [
            {
                name: `${name}__momentum`,
                tensor: state.momentum
            },
            {
                name: `${name}__hessianMoment`,
                tensor: state.hessianMoment
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
            epsilon: this.epsilon,
            seed: this.seed,
            step: this.step
        }
    }

    static get className() {
        return 'SophiaH'
    }
}

tf.serialization.registerClass(SophiaH)

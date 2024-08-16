import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './filters.js'

export default class Prodigy extends tf.Optimizer {
    constructor({
        learningRate = 1.0,
        beta1 = 0.9,
        beta2 = 0.999,
        beta3 = null,
        d = null,
        d0 = 1e-6,
        dMax = null,
        dCoef = 1.0,
        growthRate = null,
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        biasCorrection = false,
        safeguardWarmup = false,
        epsilon = 1e-8,
        step = 0
    } = {}) {
        super()
        this.learningRate = learningRate
        this.beta1 = beta1
        this.beta2 = beta2
        this.beta3 = beta3 || Math.sqrt(beta2)
        this.d = d ? d : d0
        this.d0 = d0
        this.dMax = dMax ? dMax : d0
        this.dCoef = dCoef
        this.growthRate = growthRate ? growthRate : Infinity
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.biasCorrection = biasCorrection
        this.safeguardWarmup = safeguardWarmup
        this.epsilon = epsilon
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    // Adam-style debias correction
    debias(beta, step) {
        return 1.0 - Math.pow(beta, step)
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            this.step++
            const biasCorrection = this.biasCorrection
                ? this.debias(this.beta1, this.step) /
                  Math.sqrt(this.debias(this.beta2, this.step))
                : 1.0

            const dLr = (this.d * this.learningRate) / biasCorrection

            let dNumerator = tf.scalar(0)
            let dDenom = tf.scalar(0)

            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]

                if (!this.STATE[name]) {
                    this.STATE[name] = {
                        p0: tf.variable(variable.clone()),
                        expAvg: tf.variable(tf.zerosLike(variable)),
                        expAvgSq: tf.variable(tf.zerosLike(variable)),
                        s: tf.variable(tf.zerosLike(variable))
                    }
                }

                const p0 = this.STATE[name].p0.clone()
                const expAvg = this.STATE[name].expAvg.clone()
                const expAvgSq = this.STATE[name].expAvgSq.clone()
                const s = this.STATE[name].s.clone()

                dNumerator = dNumerator.add(
                    tf
                        .dot(gradient.flatten(), tf.sub(p0, variable).flatten())
                        .mul((this.d / this.d0) * dLr)
                )

                this.STATE[name].expAvg.assign(
                    expAvg
                        .mul(this.beta1)
                        .add(gradient.mul(this.d * (1.0 - this.beta1)))
                )
                this.STATE[name].expAvgSq.assign(
                    expAvgSq
                        .mul(this.beta2)
                        .add(
                            gradient
                                .square()
                                .mul(this.d * this.d * (1.0 - this.beta2))
                        )
                )

                this.STATE[name].s.assign(
                    s
                        .mul(this.beta3)
                        .add(
                            gradient.mul(
                                (this.d / this.d0) *
                                    (this.safeguardWarmup ? this.d : dLr)
                            )
                        )
                )

                dDenom = dDenom.add(this.STATE[name].s.abs().sum())
            })

            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]
                const { expAvg, expAvgSq } = this.STATE[name]

                const denom = expAvgSq.sqrt().add(this.d * this.epsilon)

                if (
                    this.weightDecay !== 0 &&
                    !shouldExcludeFromWeightDecay(name)
                ) {
                    if (this.weightDecouple) {
                        variable.assign(
                            variable.sub(variable.mul(this.weightDecay * dLr))
                        )
                    } else if (!this.fixedDecay) {
                        gradient.assign(
                            gradient.add(variable.mul(this.weightDecay))
                        )
                    }
                }

                variable.assign(variable.sub(expAvg.div(denom).mul(dLr)))
            })

            const dHat = this.dCoef * dNumerator.div(dDenom).dataSync()[0]
            if (this.d === this.d0) this.d = Math.max(this.d, dHat)
            this.dMax = Math.max(this.dMax, dHat)
            this.d = Math.min(this.dMax, this.d * this.growthRate)
        })

        this.incrementIterations()
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).forEach(([name, state]) => {
            weights.push({ name: `${name}__p0`, tensor: state.p0 })
            weights.push({ name: `${name}__expAvg`, tensor: state.expAvg })
            weights.push({ name: `${name}__expAvgSq`, tensor: state.expAvgSq })
            weights.push({ name: `${name}__s`, tensor: state.s })
        })
        return weights
    }

    setWeights(weightValues) {
        weightValues.forEach((namedTensor) => {
            const [name, tensorName] = namedTensor.name.split('__')
            if (!this.STATE[name]) this.STATE[name] = {}
            this.STATE[name][tensorName] = tf.keep(
                tf.variable(namedTensor.tensor)
            )
        })
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            beta3: this.beta3,
            d: this.d,
            d0: this.d0,
            dMax: this.dMax,
            dCoef: this.dCoef,
            growthRate: this.growthRate,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            biasCorrection: this.biasCorrection,
            safeguardWarmup: this.safeguardWarmup,
            epsilon: this.epsilon,
            step: this.step
        }
    }

    static get className() {
        return 'Prodigy'
    }
}

tf.serialization.registerClass(Prodigy)

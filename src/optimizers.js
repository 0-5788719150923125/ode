import * as tf from '@tensorflow/tfjs'

class AdamW extends tf.AdamOptimizer {
    constructor({
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        decayRate = 1e-4
    } = {}) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.decayRate = decayRate
    }
    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Array.isArray(variableGradients)
                ? variableGradients.map((v) => v.name)
                : Object.keys(variableGradients)

            varNames.forEach((name, i) => {
                if (shouldExcludeFromWeightDecay(name)) return
                const value = this.ENGINE.registeredVariables[name]
                const newValue = tf.sub(
                    value,
                    tf.mul(this.learningRate, tf.mul(value, this.decayRate))
                )
                value.assign(newValue)
            })

            super.applyGradients(variableGradients)
        })
    }
}

class Prodigy extends tf.SGDOptimizer {
    constructor({
        learningRate = 1.0,
        beta1 = 0.9,
        beta2 = 0.999,
        beta3 = null,
        d0 = 1e-6,
        dCoef = 1.0,
        growthRate = Infinity,
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        biasCorrection = false,
        safeguardWarmup = false,
        epsilon = 1e-8
    } = {}) {
        super(learningRate)
        this.ENGINE = tf.engine()
        this.STATE = {}
        this.beta1 = beta1
        this.beta2 = beta2
        this.beta3 = beta3 || Math.sqrt(beta2)
        this.d0 = d0
        this.dCoef = dCoef
        this.growthRate = growthRate
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.biasCorrection = biasCorrection
        this.safeguardWarmup = safeguardWarmup
        this.epsilon = epsilon
        this.d = d0
        this.dMax = d0
        this.step = 1
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const beta1 = this.beta1
            const beta2 = this.beta2
            const beta3 = this.beta3
            const biasCorrection1 = 1.0 - Math.pow(beta1, this.step)
            const biasCorrection2Sq = Math.sqrt(
                1.0 - Math.pow(beta2, this.step)
            )
            const biasCorrection = this.biasCorrection
                ? biasCorrection1 / biasCorrection2Sq
                : 1.0
            const dLr = (this.d * this.learningRate) / biasCorrection

            let dNumerator = tf.scalar(0)
            let dDenom = tf.scalar(0)

            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]

                if (!this.STATE[name]) {
                    this.STATE[name] = {
                        p0: tf.keep(variable.clone()),
                        expAvg: tf.keep(tf.zerosLike(variable)),
                        expAvgSq: tf.keep(tf.zerosLike(variable)),
                        s: tf.keep(tf.zerosLike(variable))
                    }
                }

                const p0 = this.STATE[name].p0
                const expAvg = tf.clone(this.STATE[name].expAvg)
                const expAvgSq = tf.clone(this.STATE[name].expAvgSq)
                const s = tf.clone(this.STATE[name].s)

                tf.dispose([
                    this.STATE[name].expAvg,
                    this.STATE[name].expAvgSq,
                    this.STATE[name].s
                ])

                dNumerator = dNumerator.add(
                    tf
                        .dot(gradient.flatten(), tf.sub(p0, variable).flatten())
                        .mul((this.d / this.d0) * dLr)
                )

                this.STATE[name].expAvg = tf.keep(
                    expAvg.mul(beta1).add(gradient.mul(this.d * (1.0 - beta1)))
                )
                this.STATE[name].expAvgSq = tf.keep(
                    expAvgSq
                        .mul(beta2)
                        .add(
                            gradient
                                .square()
                                .mul(this.d * this.d * (1.0 - beta2))
                        )
                )

                this.STATE[name].s = tf.keep(
                    s
                        .mul(beta3)
                        .add(
                            gradient.mul(
                                (this.d / this.d0) *
                                    (this.safeguardWarmup ? this.d : dLr)
                            )
                        )
                )

                dDenom = dDenom.add(this.STATE[name].s.abs().sum())
            })

            if (dDenom.dataSync()[0] === 0) {
                return
            }

            const dHat = this.dCoef * dNumerator.div(dDenom).dataSync()[0]
            if (this.d === this.d0) {
                this.d = Math.max(this.d, dHat)
            }
            this.dMax = Math.max(this.dMax, dHat)
            this.d = Math.min(this.dMax, this.d * this.growthRate)

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
                        gradient = gradient.add(variable.mul(this.weightDecay))
                    }
                }

                variable.assign(variable.sub(expAvg.div(denom).mul(dLr)))
            })

            this.step++
        })

        super.applyGradients(variableGradients)
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            beta3: this.beta3,
            d0: this.d0,
            dCoef: this.dCoef,
            growthRate: this.growthRate,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            biasCorrection: this.biasCorrection,
            safeguardWarmup: this.safeguardWarmup,
            epsilon: this.epsilon
        }
    }
}

function shouldExcludeFromWeightDecay(name) {
    const lowerCaseName = name.toLowerCase()
    return (
        lowerCaseName.includes('norm') ||
        lowerCaseName.includes('emb') ||
        lowerCaseName.includes('bias')
    )
}

const customOptimizers = {
    AdamW: (config) => new AdamW(config),
    Prodigy: (config) => new Prodigy(config)
}

export default customOptimizers

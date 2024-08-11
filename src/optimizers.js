import * as tf from '@tensorflow/tfjs'

class AdamW extends tf.AdamOptimizer {
    constructor({
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        weightDecay = 1e-4
    } = {}) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.learningRate = learningRate
        this.weightDecay = weightDecay
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
                    tf.mul(this.learningRate, tf.mul(value, this.weightDecay))
                )
                value.assign(newValue)
            })

            super.applyGradients(variableGradients)
        })
    }

    static get className() {
        return 'AdamW'
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            epsilon: this.epsilon,
            weightDecay: this.weightDecay
        }
    }
}

tf.serialization.registerClass(AdamW)

class Lion extends tf.Optimizer {
    constructor({
        learningRate = 1e-4,
        beta1 = 0.9,
        beta2 = 0.99,
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        useGc = false,
        r = 0.95,
        adaNorm = false
    } = {}) {
        super()
        this.learningRate = learningRate
        this.beta1 = beta1
        this.beta2 = beta2
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.useGc = useGc
        this.r = r
        this.adaNorm = adaNorm
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        Object.keys(variableGradients).forEach((name) => {
            const variable = this.ENGINE.registeredVariables[name]
            let gradient = variableGradients[name]

            if (!this.STATE[name]) {
                this.STATE[name] = {
                    expAvg: tf.keep(tf.zerosLike(variable))
                }
                if (this.adaNorm) {
                    this.STATE[name].expGradNorm = tf.keep(
                        tf.variable(tf.scalar(0))
                    )
                }
            }

            tf.tidy(() => {
                if (this.useGc) {
                    const mean = gradient.mean()
                    gradient = gradient.sub(mean)
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
                        gradient = gradient.add(variable.mul(this.weightDecay))
                    }
                }

                const sGrad = this.getAdaNormGradient(gradient, name)
                const expAvg = tf.clone(this.STATE[name].expAvg)
                tf.dispose(this.STATE[name].expAvg)

                const update = expAvg
                    .clone()
                    .mul(this.beta1)
                    .add(gradient.mul(1 - this.beta1))
                    .sign()

                const updatedExpAvg = expAvg
                    .mul(this.beta2)
                    .add(sGrad.mul(1 - this.beta2))

                variable.assign(variable.sub(update.mul(this.learningRate)))

                this.STATE[name].expAvg = tf.keep(updatedExpAvg)
                tf.dispose(expAvg)
            })
        })

        this.incrementIterations()
    }

    getAdaNormGradient(gradient, name) {
        if (!this.adaNorm) {
            return gradient
        }

        const expGradNorm = this.STATE[name].expGradNorm
        const gradNorm = gradient.norm()

        expGradNorm.assign(
            expGradNorm.mul(this.r).add(gradNorm.mul(1 - this.r))
        )

        this.STATE[name].expGradNorm = tf.keep(tf.variable(expGradNorm))

        const sGrad = gradient.div(expGradNorm.maximum(1e-10))

        tf.dispose(expGradNorm)

        return sGrad
    }

    setWeights(weightValues) {
        weightValues.forEach((namedTensor) => {
            const [name, tensorName] = namedTensor.name.split('__')
            if (!this.STATE[name]) {
                this.STATE[name] = {}
            }
            this.STATE[name][tensorName] = tf.keep(
                tf.variable(namedTensor.tensor)
            )
        })
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).forEach(([name, state]) => {
            weights.push({ name: `${name}__expAvg`, tensor: state.expAvg })

            if (this.adaNorm) {
                weights.push({
                    name: `${name}__expGradNorm`,
                    tensor: state.expGradNorm
                })
            }
        })
        return weights
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            useGc: this.useGc,
            r: this.r,
            adaNorm: this.adaNorm
        }
    }

    static fromConfig(cls, config) {
        return new cls(config)
    }

    dispose() {
        Object.values(this.STATE).forEach((state) => {
            Object.values(state).forEach((tensor) => tensor.dispose())
        })
        this.STATE = {}
    }

    static get className() {
        return 'Lion'
    }
}

tf.serialization.registerClass(Lion)

class Prodigy extends tf.Optimizer {
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

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const beta1 = this.beta1
            const beta2 = this.beta2
            const beta3 = this.beta3
            const biasCorrection1 = 1.0 - Math.pow(beta1, this.step)
            const biasCorrection2 = 1.0 - Math.pow(beta2, this.step)
            const biasCorrection = this.biasCorrection
                ? biasCorrection1 / Math.sqrt(biasCorrection2)
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

class Signum extends tf.Optimizer {
    constructor({
        learningRate = 1e-3,
        momentum = 0.9,
        weightDecay = 0.0,
        weightDecouple = true
    } = {}) {
        super()
        this.learningRate = learningRate
        this.momentum = momentum
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        Object.keys(variableGradients).forEach((name) => {
            const variable = this.ENGINE.registeredVariables[name]
            let gradient = variableGradients[name]

            tf.tidy(() => {
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
                    } else {
                        gradient = gradient.add(variable.mul(this.weightDecay))
                    }
                }

                if (this.momentum > 0) {
                    if (!this.STATE[name]) {
                        this.STATE[name] = {
                            momentumBuffer: tf.keep(tf.zerosLike(variable))
                        }
                    }

                    const momentumBuffer = tf.clone(
                        this.STATE[name].momentumBuffer
                    )
                    tf.dispose(this.STATE[name].momentumBuffer)

                    const update = momentumBuffer
                        .mul(this.momentum)
                        .add(gradient.mul(1 - this.momentum))
                        .sign()

                    variable.assign(variable.sub(update.mul(this.learningRate)))

                    this.STATE[name].momentumBuffer = tf.keep(update)

                    tf.dispose(momentumBuffer)
                } else {
                    const update = gradient.sign()
                    variable.assign(variable.sub(update.mul(this.learningRate)))
                }
            })
        })

        this.incrementIterations()
    }

    setWeights(weightValues) {
        weightValues.forEach((namedTensor) => {
            const [name, tensorName] = namedTensor.name.split('__')
            if (!this.STATE[name]) {
                this.STATE[name] = {}
            }
            this.STATE[name][tensorName] = tf.keep(
                tf.variable(namedTensor.tensor)
            )
        })
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).forEach(([name, state]) => {
            weights.push({
                name: `${name}__momentumBuffer`,
                tensor: state.momentumBuffer
            })
        })
        return weights
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            momentum: this.momentum,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple
        }
    }

    static fromConfig(cls, config) {
        return new cls(config)
    }

    dispose() {
        Object.values(this.STATE).forEach((state) => {
            Object.values(state).forEach((tensor) => tensor.dispose())
        })
        this.STATE = {}
    }

    static get className() {
        return 'Signum'
    }
}

tf.serialization.registerClass(Signum)

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
    Lion: (config) => new Lion(config),
    Prodigy: (config) => new Prodigy(config),
    Signum: (config) => new Signum(config)
}

export default customOptimizers

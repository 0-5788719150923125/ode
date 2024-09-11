import * as tf from '@tensorflow/tfjs'

export default class SophiaH extends tf.Optimizer {
    constructor({
        learningRate = 6e-2,
        betas = [0.96, 0.99],
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        p = 1e-2,
        updatePeriod = 10,
        numSamples = 1,
        hessianDistribution = 'gaussian',
        epsilon = 1e-12,
        step = 1
    } = {}) {
        super()
        this.learningRate = learningRate
        this.betas = betas
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.p = p
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

                if (this.weightDecay !== 0) {
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
                    .mul(this.betas[0])
                    .add(gradient.mul(1 - this.betas[0]))

                if (this.step % this.updatePeriod === 0) {
                    const hessianEstimate = this.computeHutchinsonEstimator(
                        variable,
                        gradient
                    )
                    const hessianMoment = state.hessianMoment
                        .mul(this.betas[1])
                        .add(hessianEstimate.mul(1 - this.betas[1]))
                    state.hessianMoment.assign(hessianMoment)
                }

                const update = momentum
                    .div(
                        tf.maximum(
                            state.hessianMoment.mul(this.p),
                            this.epsilon
                        )
                    )
                    .clipByValue(-1, 1)
                variable.assign(variable.sub(update.mul(this.learningRate)))

                state.momentum.assign(momentum)
                this.STATE[name] = state
            })

            this.step++
        })
        this.incrementIterations()
    }

    computeHutchinsonEstimator(variable, gradient) {
        const u = tf.randomNormal(variable.shape)

        // Compute Hessian-vector product
        const hvp = tf
            .grad((v) => {
                return tf.sum(gradient.mul(v))
            })(variable)
            .mul(u)

        return u.mul(hvp)
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
            betas: this.betas,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            p: this.p,
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

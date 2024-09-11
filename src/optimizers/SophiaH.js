import * as tf from '@tensorflow/tfjs'

class SophiaH extends tf.Optimizer {
    constructor(config) {
        super()
        this.learningRate = config.learningRate || 6e-2
        this.betas = config.betas || [0.96, 0.99]
        this.weightDecay = config.weightDecay || 0.0
        this.weightDecouple = config.weightDecouple ?? true
        this.fixedDecay = config.fixedDecay ?? false
        this.p = config.p || 1e-2
        this.updatePeriod = config.updatePeriod || 10
        this.numSamples = config.numSamples || 1
        this.hessianDistribution = config.hessianDistribution || 'gaussian'
        this.eps = config.eps || 1e-12
        this.step = 0
        this.state = {}
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Object.keys(variableGradients)

            varNames.forEach((name) => {
                const variable = this.getVariable(name)
                const gradient = variableGradients[name]
                const state = this.state[name] || {
                    momentum: tf.zerosLike(variable),
                    hessianMoment: tf.zerosLike(variable)
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
                    .div(tf.maximum(state.hessianMoment.mul(this.p), this.eps))
                    .clipByValue(-1, 1)
                variable.assign(variable.sub(update.mul(this.learningRate)))

                state.momentum.assign(momentum)
                this.state[name] = state
            })

            this.step++
        })
        this.incrementIterations()
    }

    computeHutchinsonEstimator(variable, gradient) {
        const u = tf.variable(tf.zerosLike(variable))
        u.assign(tf.initializers[this.hessianDistribution]()(u.shape))

        const gradDotU = tf.sum(gradient.mul(u))
        const hessianVectorProduct = tf.grad((v) =>
            tf.sum(
                tf
                    .grad(() => gradDotU)(v)
                    .mul(u)
            )
        )(variable)
        return u.mul(hessianVectorProduct)
    }

    getVariable(name) {
        return (
            this.iterations.map.get(name) ||
            this.iterations.originalMap.get(name)
        )
    }

    dispose() {
        Object.values(this.state).forEach((state) => {
            state.momentum.dispose()
            state.hessianMoment.dispose()
        })
        this.state = {}
        super.dispose()
    }

    async saveIterations() {
        const stateTensors = Object.entries(this.state)
            .map(([name, state]) => [
                { name: `${name}__momentum`, tensor: state.momentum },
                { name: `${name}__hessianMoment`, tensor: state.hessianMoment }
            ])
            .flat()
        return {
            state: Object.fromEntries(
                stateTensors.map((s) => [s.name, s.tensor.arraySync()])
            ),
            step: this.step
        }
    }

    async getWeights() {
        const stateEntries = await Promise.all(
            Object.entries(this.state)
                .map(async ([name, state]) => [
                    {
                        name: `${name}__momentum`,
                        tensor: await state.momentum.data()
                    },
                    {
                        name: `${name}__hessianMoment`,
                        tensor: await state.hessianMoment.data()
                    }
                ])
                .flat()
        )
        return {
            state: Object.fromEntries(
                stateEntries.map((s) => [s.name, s.tensor])
            ),
            step: this.step
        }
    }

    async setWeights(weightValues) {
        weightValues.state &&
            Object.entries(weightValues.state).forEach(([name, tensor]) => {
                const [varName, stateKey] = name.split('__')
                if (!this.state[varName]) {
                    this.state[varName] = {}
                }
                this.state[varName][stateKey] = tf.tensor(tensor)
            })
        this.step = weightValues.step
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
            eps: this.eps
        }
    }

    static get className() {
        return 'SophiaH'
    }
}

tf.serialization.registerClass(SophiaH)

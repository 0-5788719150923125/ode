import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './_ops.js'

export default class Lion extends tf.Optimizer {
    constructor({
        learningRate = 1e-4,
        beta1 = 0.9,
        beta2 = 0.99,
        weightDecay = 0.0,
        weightDecouple = true,
        fixedDecay = false,
        useGc = false,
        r = 0.95,
        adaNorm = false,
        step = 1
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
        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        Object.keys(variableGradients).forEach((name) => {
            const variable = this.ENGINE.registeredVariables[name]
            let gradient = variableGradients[name]

            if (!this.STATE[name]) {
                this.STATE[name] = {
                    expAvg: tf.variable(tf.zerosLike(variable))
                }
                if (this.adaNorm) {
                    this.STATE[name].expGradNorm = tf.variable(tf.scalar(0))
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
                const expAvg = this.STATE[name].expAvg

                const update = expAvg
                    .clone()
                    .mul(this.beta1)
                    .add(gradient.mul(1 - this.beta1))
                    .sign()

                const updatedExpAvg = expAvg
                    .mul(this.beta2)
                    .add(sGrad.mul(1 - this.beta2))

                variable.assign(variable.sub(update.mul(this.learningRate)))

                this.STATE[name].expAvg.assign(updatedExpAvg)
            })
        })

        this.incrementIterations()
        this.step++
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

        this.STATE[name].expGradNorm.assign(expGradNorm)

        const sGrad = gradient.div(expGradNorm.maximum(1e-10))

        return sGrad
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
            adaNorm: this.adaNorm,
            step: this.step
        }
    }

    static get className() {
        return 'Lion'
    }
}

tf.serialization.registerClass(Lion)

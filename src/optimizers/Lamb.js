import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './_ops.js'

export default class Lamb extends tf.Optimizer {
    constructor({
        learningRate = 1e-3,
        betas = [0.9, 0.999],
        weightDecay = 0,
        weightDecouple = true,
        fixedDecay = false,
        rectify = false,
        maxGradNorm = 1.0,
        eps = 1e-6,
        preNorm = false,
        adaNorm = false,
        r = 0.95,
        step = 1
    } = {}) {
        super()

        this.learningRate = learningRate
        this.betas = betas
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.rectify = rectify
        this.eps = eps
        this.preNorm = preNorm
        this.adaNorm = adaNorm
        this.r = r
        this.maxGradNorm = maxGradNorm
        this.clamp = 10.0

        this.step = step
        this.ENGINE = tf.engine()
        this.STATE = {}
    }

    applyGradients(variableGradients) {
        if (this.preNorm) {
            var globalGradNorm = this.getGlobalGradientNorm(variableGradients)
        }

        const lr = this.learningRate
        const [beta1, beta2] = this.betas

        const beta3 = 1 - beta1
        const biasCorrection1 = 1 - Math.pow(beta1, this.step)
        const biasCorrection2 = 1 - Math.pow(beta2, this.step)

        Object.entries(variableGradients).forEach(([name, gradient]) => {
            const variable = this.ENGINE.registeredVariables[name]

            if (!this.STATE[name]) {
                this.STATE[name] = {
                    expAvg: tf.variable(tf.zerosLike(variable)),
                    expAvgSq: tf.variable(tf.zerosLike(variable))
                }
                if (this.adaNorm) {
                    this.STATE[name].expGradNorm = tf.variable(tf.scalar(0))
                }
            }

            tf.tidy(() => {
                const { expAvg, expAvgSq } = this.STATE[name]

                if (this.preNorm) {
                    gradient = gradient.div(globalGradNorm)
                }

                if (
                    this.weightDecay !== 0 &&
                    !shouldExcludeFromWeightDecay(name)
                ) {
                    if (this.weightDecouple) {
                        variable.assign(
                            variable.sub(variable.mul(this.weightDecay).mul(lr))
                        )
                    } else if (!this.fixedDecay) {
                        gradient = gradient.add(variable.mul(this.weightDecay))
                    }
                }

                const sGrad = this.getAdaNormGradient(gradient, name)

                expAvg.assign(expAvg.mul(beta1).add(sGrad.mul(beta3)))
                expAvgSq.assign(
                    expAvgSq.mul(beta2).add(gradient.square().mul(1 - beta2))
                )

                const expAvgCorrected = expAvg.div(biasCorrection1)
                const expAvgSqCorrected = expAvgSq.div(biasCorrection2)

                let update
                if (this.rectify) {
                    const nSmaMax = 2.0 / (1.0 - beta2) - 1.0
                    const beta2t = Math.pow(beta2, this.step)
                    const nSma =
                        nSmaMax - (2 * this.step * beta2t) / (1.0 - beta2t)

                    let rt
                    if (nSma >= 5) {
                        // n_sma_threshold
                        rt = Math.sqrt(
                            ((((((1.0 - beta2t) * (nSma - 4)) / (nSmaMax - 4)) *
                                (nSma - 2)) /
                                nSma) *
                                nSmaMax) /
                                (nSmaMax - 2)
                        )
                    } else if (this.degenerated_to_sgd) {
                        rt = 1.0
                    } else {
                        rt = -1.0
                    }

                    const stepSize = lr * rt

                    if (nSma >= 5) {
                        // n_sma_threshold
                        const deNom = expAvgSqCorrected.sqrt().add(this.eps)
                        update = expAvgCorrected.div(deNom).mul(stepSize)
                    } else {
                        update = expAvgCorrected.mul(stepSize)
                    }
                } else {
                    update = expAvgCorrected.div(
                        expAvgSqCorrected.sqrt().add(this.eps)
                    )
                }

                const weightNorm = variable
                    .norm()
                    .maximum(0)
                    .minimum(this.clamp)
                const updateNorm = update.norm()
                let trustRatio = weightNorm
                    .div(updateNorm.add(this.eps))
                    .dataSync()[0]

                if (isNaN(trustRatio)) trustRatio = 1

                variable.assign(variable.sub(update.mul(lr * trustRatio)))
            })
        })

        this.step++

        return this.incrementIterations()
    }

    getAdaNormGradient(gradient, name) {
        if (!this.adaNorm) return gradient

        let { expGradNorm } = this.STATE[name]

        expGradNorm.assign(
            expGradNorm.mul(this.r).add(gradient.norm().mul(1 - this.r))
        )
        return gradient.div(expGradNorm.add(this.eps))
    }

    getGlobalGradientNorm(variableGradients) {
        if (this.maxGradNorm === 0) return 1.0

        const globalNorm = tf.sqrt(
            Object.values(variableGradients)
                .map((gradient) => gradient.square().sum())
                .reduce((acc, curr) => acc.add(curr))
        )

        return globalNorm
            .maximum(this.maxGradNorm)
            .div(this.maxGradNorm)
            .add(this.eps)
    }

    getWeights() {
        const weights = []
        Object.entries(this.STATE).map(([name, state]) => {
            weights.push({ name: `${name}__expAvg`, tensor: state.expAvg })
            weights.push({ name: `${name}__expAvgSq`, tensor: state.expAvgSq })
            if (state.expGradNorm) {
                weights.push({
                    name: `${name}__expGradNorm`,
                    tensor: state.expGradNorm
                })
            }
        })
        return weights
    }

    setWeights(weights) {
        weights.forEach(({ name, tensor }) => {
            const [varName, property] = name.split('__')
            if (!this.STATE[varName]) this.STATE[varName] = {}
            this.STATE[varName][property] = tf.variable(tensor)
        })
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            betas: this.betas,
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            rectify: this.rectify,
            eps: this.eps,
            preNorm: this.preNorm,
            adaNorm: this.adaNorm,
            r: this.r,
            maxGradNorm: this.maxGradNorm,
            step: this.step
        }
    }

    static get className() {
        return 'Lamb'
    }
}

tf.serialization.registerClass(Lamb)

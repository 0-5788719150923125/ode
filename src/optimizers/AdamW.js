import * as tf from '@tensorflow/tfjs'
import { applyWeightDecay } from './_ops.js'

export default class AdamW extends tf.AdamOptimizer {
    constructor({
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        weightDecay = 1e-4,
        weightDecouple = true,
        fixedDecay = false,
        step = 1
    } = {}) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.learningRate = learningRate
        this.weightDecay = weightDecay
        this.weightDecouple = weightDecouple
        this.fixedDecay = fixedDecay
        this.step = step
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            Object.entries(variableGradients).forEach(([name, gradient]) => {
                const variable = this.ENGINE.registeredVariables[name]
                gradient = applyWeightDecay(
                    variable,
                    gradient,
                    name,
                    this.learningRate,
                    this.weightDecay,
                    this.weightDecouple,
                    this.fixedDecay
                )
            })
            super.applyGradients(variableGradients)
        })

        this.step++
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
            weightDecay: this.weightDecay,
            weightDecouple: this.weightDecouple,
            fixedDecay: this.fixedDecay,
            step: this.step
        }
    }
}

tf.serialization.registerClass(AdamW)

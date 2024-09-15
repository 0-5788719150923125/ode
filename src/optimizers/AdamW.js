import * as tf from '@tensorflow/tfjs'
import { applyWeightDecay } from './_ops.js'

export default class AdamW extends tf.AdamOptimizer {
    constructor({
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        weightDecay = 1e-4,
        step = 1
    } = {}) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.learningRate = learningRate
        this.weightDecay = weightDecay
        this.step = step
    }

    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Array.isArray(variableGradients)
                ? variableGradients.map((v) => v.name)
                : Object.keys(variableGradients)

            varNames.forEach((name, i) => {
                if (shouldExcludeFromWeightDecay(name)) return
                const variable = this.ENGINE.registeredVariables[name]
                let gradient = variableGradients[name]
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
            step: this.step
        }
    }
}

tf.serialization.registerClass(AdamW)

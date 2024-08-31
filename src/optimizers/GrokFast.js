import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './_ops.js'

// https://arxiv.org/abs/2405.20233v2
export default class GrokFast extends tf.AdamOptimizer {
    constructor({
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        weightDecay = 1e-4,
        alpha = 0.98,
        lamb = 2.0,
        step = 1
    } = {}) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.learningRate = learningRate
        this.weightDecay = weightDecay
        this.alpha = alpha
        this.lamb = lamb
        this.step = step
        this.grads = {}
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

                // Apply GrokFast EMA gradient filter
                const grad = variableGradients[name]
                if (this.grads[name] == null) {
                    this.grads[name] = tf.variable(tf.zerosLike(grad))
                }
                this.grads[name].assign(
                    tf.add(
                        tf.mul(this.grads[name], this.alpha),
                        tf.mul(grad, 1 - this.alpha)
                    )
                )
                const filteredGrad = tf.add(
                    grad,
                    tf.mul(this.grads[name], this.lamb)
                )

                tf.dispose([variableGradients[name]])

                variableGradients[name] = filteredGrad
            })

            super.applyGradients(variableGradients)
        })

        this.step++
    }

    static get className() {
        return 'AdamWGrokfast'
    }

    getConfig() {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            epsilon: this.epsilon,
            weightDecay: this.weightDecay,
            alpha: this.alpha,
            lamb: this.lamb,
            step: this.step
        }
    }
}

tf.serialization.registerClass(GrokFast)

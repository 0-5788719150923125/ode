import * as tf from '@tensorflow/tfjs'
import { shouldExcludeFromWeightDecay } from './filters.js'

export default class Signum extends tf.Optimizer {
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

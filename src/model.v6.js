import ODE from './model.v4.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super({ ...config, learningRate: 5e-4, weightDecay: 5e-5 })
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.ConstantScheduler({
                max: this.config.learningRate,
                warmupSteps: this.config.warmupSteps
            })
        ]
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay,
                beta1: 0.95,
                beta2: 0.98,
                useGc: true,
                adaNorm: true
            })
        ]
    }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.SophiaH({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay
    //         })
    //     ]
    // }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.AdamW({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay
    //         })
    //     ]
    // }

    // defineReductionLayer() {
    //     return this.ode.layers.AttentionBasedReduction({
    //         units: this.config.units,
    //         hiddenDim: this.config.headDim
    //     })
    // }

    // defineLossFunction() {
    //     return {
    //         name: 'MiLeCrossEntropy',
    //         reduction: this.tf.Reduction.MEAN
    //     }
    // }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.Prodigy({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay,
    //             safeguardWarmup: true,
    //             biasCorrection: true
    //         })
    //     ]
    // }

    // defineSchedulers() {
    //     return [
    //         this.ode.schedulers.cosineScheduler(
    //             0,
    //             this.config.learningRate,
    //             this.config.trainSteps,
    //             this.config.warmupSteps
    //         )
    //     ]
    // }

    // defineReductionLayer() {
    //     return this.ode.layers.dense({
    //         prefix: 'op',
    //         units: this.config.units,
    //         kernelInitializer: this.ode.initializers.glorotUniform()
    //     })
    // }
}

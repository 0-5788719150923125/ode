import ODE from './model.v4.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super({ learningRate: 0.0008, weightDecay: 1e-2, ...config })
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.ConstantScheduler({
                max: this.config.learningRate,
                warmupSteps: this.config.warmupSteps
            })
        ]
    }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.SophiaH({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay,
    //             updatePeriod: 9,
    //             numSamples: 1,
    //             seed: this.config.seed
    //         })
    //     ]
    // }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.AdamG({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay
    //         })
    //     ]
    // }

    // defineOptimizers() {
    //     return [
    //         this.ode.optimizers.Lion({
    //             learningRate: this.config.learningRate,
    //             weightDecay: this.config.weightDecay,
    //             useGc: true,
    //             adaNorm: true
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

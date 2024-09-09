import ODE from './model.v4.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        const defaults = {
            layers: 6,
            units: 180,
            embeddings: 540,
            numHeads: 4,
            queriesPerHead: 2,
            headDim: 45,
            mlpDim: 1080,
            useBias: true,
            ALiBiLength: 1024,
            learningRate: 1e-4,
            weightDecay: 1e-5,
            warmupSteps: 128
        }
        super({ ...defaults, ...config })
    }

    defineReductionLayer() {
        return this.ode.layers.LowRankFactorization({
            units: this.config.units,
            rank: 90
        })
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

    // defineLossFunction() {
    //     return {
    //         name: 'softmaxCrossEntropy',
    //         reduction: this.tf.Reduction.MEAN
    //     }
    // }

    // defineReductionLayer() {
    //     return this.ode.layers.dense({
    //         prefix: 'op',
    //         units: this.config.units,
    //         kernelInitializer: this.ode.initializers.glorotUniform()
    //     })
    // }
}

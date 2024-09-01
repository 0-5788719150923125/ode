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

    defineSchedulers() {
        return [
            this.ode.schedulers.constantScheduler(
                this.config.learningRate,
                this.config.warmupSteps
            )
        ]
    }

    defineReductionLayer() {
        return this.ode.layers.dense({
            prefix: 'op',
            units: this.config.units,
            kernelInitializer: this.ode.initializers.glorotUniform()
        })
    }
}

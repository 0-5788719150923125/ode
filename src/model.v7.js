import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    defineReductionLayer() {
        return this.ode.layers.dense({
            prefix: 'op',
            units: this.config.units,
            kernelInitializer: this.ode.initializers.glorotUniform()
        })
    }
}

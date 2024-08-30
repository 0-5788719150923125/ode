import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        super(config)
        this.config.selfModel = true
        this.config.auxiliaryWeight = 10.0
    }
}

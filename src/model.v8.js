import ODE from './model.v7.js'

/**
 * In development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        super(config)
        this.selfModel = true
        this.auxiliaryWeight = 0.1
    }
}

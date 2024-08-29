import ODE from './model.v7.js'

/**
 * In development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        const defaults = {
            selfModel: true,
            auxiliaryWeight: 0.1
        }
        super({ ...defaults, ...config })
    }
}

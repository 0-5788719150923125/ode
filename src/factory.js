import v0 from './model.v0.js'
import v1 from './model.v1.js'
import v2 from './model.v2.js'
import v3 from './model.v3.js'
import v4 from './model.v4.js'

export default async function loadModelVersion(args) {
    const defaults = {
        version: 4,
        ...args
    }
    const models = [v0, v1, v2, v3, v4]
    try {
        const module = models[Number(defaults.version)]
        return new module(args)
    } catch (error) {
        console.error(
            `Failed to load model version ${defaults.version}:`,
            error
        )
        throw error
    }
}

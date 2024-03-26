import './model.v0.js'
import './model.v1.js'
import './model.v2.js'
import './model.v3.js'
import './model.v4.js'

export default async function loadModelVersion(args) {
    const defaults = {
        version: 4,
        ...args
    }
    try {
        const module = await import(`./model.v${defaults.version}.js`)
        return new module.default(args)
    } catch (error) {
        console.error(
            `Failed to load model version ${defaults.version}:`,
            error
        )
        throw error
    }
}

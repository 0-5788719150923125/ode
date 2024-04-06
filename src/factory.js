import v0 from './model.v0.js'
import v1 from './model.v1.js'
import v2 from './model.v2.js'
import v3 from './model.v3.js'
import v4 from './model.v4.js'
import v5 from './model.v5.js'
import v6 from './model.v6.js'
import v7 from './model.v7.js'

const defaultVersions = [v0, v1, v2, v3, v4, v5, v6, v7]

export default async function loadModelVersion(args) {
    const defaults = {
        version: 4,
        ...args
    }
    try {
        // const module = await import(`./model.v${defaults.version}.js`)
        // return new module.default(args)
        return defaultVersions[defaults.version]
    } catch (error) {
        console.error(
            `Failed to load model version ${defaults.version}:`,
            error
        )
        throw error
    }
}

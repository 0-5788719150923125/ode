import v0 from './model.v0.js'
import v1 from './model.v1.js'
import v2 from './model.v2.js'
import v3 from './model.v3.js'
import v4 from './model.v4.js'
import v5 from './model.v5a.js'
import v6 from './model.v6.js'
import v7 from './model.v7.js'
// import v8 from './model.v8.js'
// import v9 from './model.v9.js'

const modules = [
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7
    // v8
    // v9
]

export default async function loadModelVersion(args) {
    const config = {
        backend: 'cpu',
        version: 3,
        ...args
    }
    try {
        return new modules[config.version](args)
    } catch (error) {
        console.error(`Failed to load model version ${config.version}:`, error)
        throw error
    }
}

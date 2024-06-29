import * as tfjs from '@tensorflow/tfjs'
import customLayers from './layers.js'

class ExpertBase {
    constructor(config) {
        this.tf = tfjs
        this.ode = {
            layers: customLayers
        }
        this.expertArgs = config || {
            type: 'SelfAttention',
            projection: 64
        }
        this.expertType = this.expertArgs.type
        this.inputShape = this.expertArgs.inputShape
        this.defineExpert()
    }

    defineExpert() {
        const inputs = this.ode.layers.input({
            shape: this.inputShape
        })

        const layer = this.findLayerByType(this.expertType)(this.expertArgs)

        const outputs = layer.apply(inputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    findLayerByType(key) {
        const lowercaseKey = key.toLowerCase()
        const match = Object.keys(this.ode.layers).find(
            (k) => k.toLowerCase() === lowercaseKey
        )
        return match ? this.ode.layers[match] : undefined
    }

    // async load(type = 'file', path = `data/models/ode/model.json`) {
    //     await this.preInit()
    //     this.model = await this.tf.loadLayersModel(`${type}://${path}`, {
    //         strict: true,
    //         streamWeights: true
    //     })
    //     console.log('successfully loaded model from disk')
    //     this.defineSchedulers()
    //     this.postInit()
    // }

    // async save(type = 'file', path = `data/models/ode`) {
    //     if (type === 'file') {
    //         const fs = await import('fs')
    //         fs.mkdirSync(path, { recursive: true })
    //     }
    //     await this.model.save(`${type}://${path}`, { includeOptimizer: true })
    // }
}

const expertHandler = (config) => new ExpertBase(config)
export default expertHandler

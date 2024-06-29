import * as tf from '@tensorflow/tfjs'
import customLayers from './layers.js'

class ExpertBase {
    constructor(config) {
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
        return this.model
    }

    defineExpert() {
        const inputs = this.ode.layers.input({
            shape: this.inputShape
        })

        const layer = this.findLayerByType(this.expertType)(this.expertArgs)

        const outputs = layer.apply(inputs)

        this.model = tf.model({ inputs, outputs })
    }

    findLayerByType(key) {
        const lowercaseKey = key.toLowerCase()
        const match = Object.keys(this.ode.layers).find(
            (k) => k.toLowerCase() === lowercaseKey
        )
        return match ? this.ode.layers[match] : undefined
    }
}

const expertHandler = (config) => new ExpertBase(config)
export default expertHandler

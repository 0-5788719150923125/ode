import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

// Originally adapted from:
// https://gist.githubusercontent.com/BenjaminWegener/311292080a71becbe5a8c0cc7657657d/raw/fd4f1f96184b58dace1854d0440d8c9dde3fd712/attention_layer_tfjs
export default class LambdaLayer extends LayerBase {
    constructor(config) {
        super(config)
        if (config.name === undefined) {
            config.name = (+new Date() * Math.random()).toString(36)
        }
        this.name = config.name
        this.lambdaFunction = config.lambdaFunction
        this.lambdaOutputShape = config.lambdaOutputShape
    }
    call(input) {
        return tf.tidy(() => {
            let result = null
            eval(this.lambdaFunction)
            return result
        })
    }
    computeOutputShape(inputShape) {
        if (this.lambdaOutputShape === undefined) {
            //if no outputshape provided, try to set as inputshape
            return inputShape[0]
        } else {
            return this.lambdaOutputShape
        }
    }
    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            lambdaFunction: this.lambdaFunction,
            lambdaOutputShape: this.lambdaOutputShape
        })
        return config
    }
}

tf.serialization.registerClass(LambdaLayer)

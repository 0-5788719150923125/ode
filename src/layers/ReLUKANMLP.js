import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class ReLUKANMLP extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 1024
        this.width = config.width || [128, 256, 512]
        this.grid = config.grid || 10
        this.k = config.k || 3
        this.trainAB = config.trainAB || true
    }

    build(inputShape) {
        const lastDim = inputShape[inputShape.length - 1]
        this.rkLayers = []
        for (let i = 0; i < this.width.length; i++) {
            const phaseLow = tf.range(-this.k, this.grid).div(this.grid)
            const phaseHeight = phaseLow.add((this.k + 1) / this.grid)

            this.rkLayers.push({
                phaseLow: this.addWeight(
                    `phaseLow_${i}`,
                    [lastDim, this.grid + this.k],
                    'float32',
                    this.initializers.constant({ value: phaseLow }),
                    this.trainAB
                ),
                phaseHeight: this.addWeight(
                    `phaseHeight_${i}`,
                    [lastDim, this.grid + this.k],
                    'float32',
                    this.initializers.constant({ value: phaseHeight }),
                    this.trainAB
                ),
                equalSizeConvKernel: this.addWeight(
                    `equalSizeConvKernel_${i}`,
                    [this.grid + this.k, lastDim, this.width[i]],
                    'float32',
                    this.initializers.glorotNormal()
                )
            })
        }
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            let x = Array.isArray(inputs) ? inputs[0] : inputs
            const inputShape = x.shape
            const lastDim = inputShape[inputShape.length - 1]
            x = tf.reshape(x, [-1, lastDim])

            for (let i = 0; i < this.rkLayers.length; i++) {
                const rkLayer = this.rkLayers[i]
                const x1 = tf.relu(x.sub(rkLayer.phaseLow.read()))
                const x2 = tf.relu(rkLayer.phaseHeight.read().sub(x))
                let y = x1
                    .mul(x2)
                    .mul(
                        (4 * this.grid * this.grid) /
                            ((this.k + 1) * (this.k + 1))
                    )
                y = y.square()
                y = y.reshape([y.shape[0], 1, this.grid + this.k, lastDim])
                y = tf.conv2d(
                    y,
                    rkLayer.equalSizeConvKernel.read(),
                    [1, 1],
                    'valid'
                )
                x = y.reshape([y.shape[0], this.width[i]])
            }

            const outputShape = inputShape
                .slice(0, -1)
                .concat([this.width[this.width.length - 1]])
            x = tf.reshape(x, outputShape)

            return x
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            width: this.width,
            grid: this.grid,
            k: this.k,
            trainAB: this.trainAB
        }
    }
}

tf.serialization.registerClass(ReLUKANMLP)

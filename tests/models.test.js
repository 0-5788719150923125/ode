import { jest, beforeAll, afterAll, describe, it, expect } from '@jest/globals'
import * as tf from '@tensorflow/tfjs'
import ODE from '../src/index.js'

// Suppress console output
const suppressAllConsoleLogs = () => {
    jest.spyOn(console, 'log').mockImplementation(() => {})
    jest.spyOn(console, 'info').mockImplementation(() => {})
    jest.spyOn(console, 'warn').mockImplementation(() => {})
    // jest.spyOn(console, 'error').mockImplementation(() => {})
}

beforeAll(async () => {
    suppressAllConsoleLogs()
})

describe('Model tests', () => {
    const models = [1, 2, 3, 4, 6, 7]
    const timeout = 360_000

    models.forEach((version) => {
        describe(`Model version ${version}`, () => {
            let net

            beforeAll(async () => {
                net = await ODE({
                    backend: 'tensorflow',
                    version,
                    sampleLength: 64,
                    contextLength: 64
                })
            }, timeout)

            afterAll(async () => {
                // Clean up if necessary
                if (net && typeof net.cleanup === 'function') {
                    await net.cleanup()
                }
                net = null
            })

            it('is initialized', async () => {
                await net.init()
            })

            it('contains a TFJS model', () => {
                expect(net.model).toBeInstanceOf(tf.LayersModel)
            })

            it(
                'can step',
                async () => {
                    const dataSampler = net.ode.samplers.HTTPSampler()
                    await net.train(dataSampler, {
                        batchSize: 1,
                        gradientAccumulationSteps: 3,
                        sampleLength: 64,
                        trainSteps: 3
                    })
                },
                timeout
            )

            it(
                'can infer',
                async () => {
                    const output = await net.generate({
                        prompt: 'Once upon a time, ',
                        doSample: true,
                        temperature: 0.7,
                        maxNewTokens: 16,
                        repetitionPenalty: 1.2
                    })
                    expect(typeof output).toBe('string')
                    expect(output.startsWith('Once upon a time')).toBe(true)
                },
                timeout
            )
        })
    })
})

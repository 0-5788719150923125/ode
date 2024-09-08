import { jest } from '@jest/globals'
import tokenizers from '../src/tokenizers.js'
import samplers from '../src/samplers.js'

const suppressAllConsoleLogs = () => {
    jest.spyOn(console, 'log').mockImplementation(() => {})
    jest.spyOn(console, 'info').mockImplementation(() => {})
    jest.spyOn(console, 'warn').mockImplementation(() => {})
}

beforeAll(async () => {
    suppressAllConsoleLogs()
})

const text =
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

// Load a tokenizer
let tokenizer = tokenizers.TokenMonster({
    model: 'englishcode-4096-consistent-v1'
})
beforeAll(async () => {
    await tokenizer.init()
})

const config = {
    tokenizer: tokenizer,
    maxSeqLen: 64,
    isValidating: false
}

describe('cosmopedia sampler', () => {
    it('can sample at random', async () => {
        const sampler = samplers.CosmopediaSampler()
        const data = await sampler.take(config)
        expect(typeof data[0] === 'number').toBe(true)
        expect(data[19] !== 15 || data[20] !== 645).toBe(true)
    })

    it('can sample with a seed', async () => {
        const sampler = samplers.CosmopediaSampler({ seed: 42 })
        const data = await sampler.take(config)
        expect(data[0] === 438 && data[1] === 2088).toBe(true)
    })

    it('returns data with the correct length', async () => {
        const sampler = samplers.CosmopediaSampler()
        const data = await sampler.take(config)
        expect(data.length === config.maxSeqLen).toBe(true)
    })

    it('can stride', async () => {
        const sampler = samplers.CosmopediaSampler({ seed: 42, stride: 2 })
        const firstBatch = await sampler.take(config)
        const secondBatch = await sampler.take(config)
        expect(
            firstBatch
                .slice(-2)
                .every((v, i) => v === secondBatch.slice(0, 2)[i])
        ).toBe(true)
    })
})

describe('phi sampler', () => {
    it('can sample at random', async () => {
        const sampler = samplers.PhiSampler()
        const data = await sampler.take(config)
        expect(typeof data[0] === 'number').toBe(true)
        expect(data[19] !== 647 || data[20] !== 36).toBe(true)
    })

    it('can sample with a seed', async () => {
        const sampler = samplers.PhiSampler({ seed: 42 })
        const data = await sampler.take(config)
        expect(data[0] === 431 && data[1] === 1).toBe(true)
    })

    it('returns data with the correct length', async () => {
        const sampler = samplers.PhiSampler()
        const data = await sampler.take(config)
        expect(data.length === config.maxSeqLen).toBe(true)
    })

    it('can stride', async () => {
        const sampler = samplers.PhiSampler({ seed: 42, stride: 2 })
        const firstBatch = await sampler.take(config)
        const secondBatch = await sampler.take(config)
        expect(
            firstBatch
                .slice(-2)
                .every((v, i) => v === secondBatch.slice(0, 2)[i])
        ).toBe(true)
    })
})

describe('weighted sampler', () => {
    const rates = [1.0, 0.5]
    const sampler = samplers.WeightedSampler({
        samplers: [
            samplers.CosmopediaSampler({ seed: 42, stride: 2 }),
            samplers.PhiSampler({ seed: 42, stride: 2 })
        ],
        rates: rates
    })

    it('can sample with a seed', async () => {
        const data = await sampler.take(config)
        expect(data[0] === 438 && data[1] === 2088).toBe(true)
    })

    it('returns data with the correct length', async () => {
        const data = await sampler.take(config)
        expect(data.length === config.maxSeqLen).toBe(true)
    })

    // it('can stride', async () => {
    //     const sampler = samplers.PhiSampler({ seed: 42, stride: 2 })
    //     const firstBatch = await sampler.take(config)
    //     const secondBatch = await sampler.take(config)
    //     expect(
    //         firstBatch
    //             .slice(-2)
    //             .every((v, i) => v === secondBatch.slice(0, 2)[i])
    //     ).toBe(true)
    // })
})

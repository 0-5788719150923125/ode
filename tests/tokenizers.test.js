import { jest } from '@jest/globals'
import tokenizers from '../src/tokenizers.js'

const suppressAllConsoleLogs = () => {
    jest.spyOn(console, 'log').mockImplementation(() => {})
    jest.spyOn(console, 'info').mockImplementation(() => {})
    jest.spyOn(console, 'warn').mockImplementation(() => {})
}

beforeAll(async () => {
    suppressAllConsoleLogs()
})

describe('Character', () => {
    const tokenizer = tokenizers.CharacterTokenizer()

    it('can encode and decode text', async () => {
        const text = 'Hello, world!'
        const encoded = tokenizer.encode(text)
        const decoded = tokenizer.decode(encoded)
        expect(text === decoded).toBe(true)
    })

    it('has correct length', async () => {
        expect(tokenizer.getLength()).toBe(374)
    })
})

describe('TokenMonster', () => {
    const tokenizer = tokenizers.TokenMonster({
        model: 'englishcode-4096-consistent-v1'
    })

    beforeAll(async () => {
        await tokenizer.init()
    })

    it('can encode and decode text', async () => {
        const text = 'Hello, world!'
        const encoded = tokenizer.encode(text)
        const decoded = tokenizer.decode(encoded)
        expect(text === decoded).toBe(true)
    })

    it('has correct length', async () => {
        expect(tokenizer.getLength()).toBe(4096)
    })
})

describe('Xenova', () => {
    const tokenizer = tokenizers.XenovaTokenizer({
        model: 'openai-community/gpt2'
    })

    beforeAll(async () => {
        await tokenizer.init()
    })

    it('can encode and decode text', async () => {
        const text = 'Hello, world!'
        const encoded = tokenizer.encode(text)
        const decoded = tokenizer.decode(encoded)
        expect(text === decoded).toBe(true)
    })

    it('has correct length', async () => {
        expect(tokenizer.getLength()).toBe(50257)
    })
})

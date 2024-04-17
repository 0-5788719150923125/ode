// import pako from 'pako'
import { shaks13 } from './data.js'

class TokenizerBase {
    constructor() {
        // pass
    }

    getLength() {
        return this.vocab.length
    }

    encode(string) {
        // not implemented
    }

    decode(array) {
        // not implemented
    }

    writeVocabularyToFile(path) {
        // skip
    }
}

// class CompressedBinaryTokenizer {
//     constructor({
//         minLength = 1,
//         maxLength = 23,
//         corpus = '',
//         chunkSize = 1024 * 256,
//         maxVocabLength = 10000
//     } = {}) {
//         this.patternFrequency = new Map() // Tracks pattern frequencies
//         this.vocab = new Map() // Maps binary pattern to index
//         this.indexToPattern = new Map() // Maps index to binary pattern for decoding
//         this.minLength = minLength
//         this.maxLength = maxLength
//         this.corpus = corpus
//         this.chunkSize = chunkSize
//         this.maxVocabLength = maxVocabLength
//         this.train()
//         this.finalizeVocabulary()
//     }

//     train() {
//         const encoder = new TextEncoder()
//         let mb = 0
//         for (
//             let start = 0;
//             start < this.corpus.length;
//             start += this.chunkSize
//         ) {
//             console.log(
//                 `chunking: ${mb / 4}mb, length: ${this.patternFrequency.size}`
//             )
//             mb++
//             const end = Math.min(start + this.chunkSize, this.corpus.length)
//             const chunk = this.corpus.substring(start, end)
//             const compressed = pako.deflate(encoder.encode(chunk))
//             let binaryString = ''
//             compressed.forEach((byte) => {
//                 binaryString += byte.toString(2).padStart(8, '0')
//             })

//             for (
//                 let length = this.minLength;
//                 length <= this.maxLength;
//                 length++
//             ) {
//                 for (let i = 0; i <= binaryString.length - length; i++) {
//                     const pattern = binaryString.substring(i, i + length)
//                     this.patternFrequency.set(
//                         pattern,
//                         (this.patternFrequency.get(pattern) || 0) + 1
//                     )
//                 }
//             }
//         }
//     }

//     finalizeVocabulary() {
//         // Sort patterns by frequency and keep only the top `maxVocabLength` patterns
//         const sortedPatterns = Array.from(this.patternFrequency.entries())
//             .sort((a, b) => b[1] - a[1])
//             .slice(0, this.maxVocabLength)

//         // Clear patternFrequency to free memory
//         this.patternFrequency.clear()

//         // Assign indices to the most common patterns
//         sortedPatterns.forEach(([pattern], index) => {
//             this.vocab.set(pattern, index)
//             this.indexToPattern.set(index, pattern)
//         })
//     }

//     encode(string) {
//         const encoder = new TextEncoder()
//         const compressed = pako.deflate(encoder.encode(string))
//         let binaryString = ''
//         compressed.forEach((byte) => {
//             binaryString += byte.toString(2).padStart(8, '0')
//         })

//         const tokens = []
//         for (let i = 0; i < binaryString.length; ) {
//             let tokenFound = false
//             for (
//                 let length = this.maxLength;
//                 length >= this.minLength;
//                 length--
//             ) {
//                 if (i + length > binaryString.length) continue
//                 const potentialToken = binaryString.substring(i, i + length)
//                 if (this.vocab.has(potentialToken)) {
//                     tokens.push(this.vocab.get(potentialToken))
//                     i += length
//                     tokenFound = true
//                     break
//                 }
//             }
//             if (!tokenFound) i++
//         }
//         return tokens
//     }

//     decode(tokens) {
//         let binaryString = ''
//         tokens.forEach((index) => {
//             const pattern = this.indexToPattern.get(index)
//             if (pattern) {
//                 binaryString += pattern
//             } else {
//                 // Ignore invalid tokens by not adding anything to binaryString
//             }
//         })

//         try {
//             const byteArray = []
//             for (let i = 0; i < binaryString.length; i += 8) {
//                 byteArray.push(parseInt(binaryString.slice(i, i + 8), 2))
//             }
//             const decompressed = pako.inflate(new Uint8Array(byteArray))
//             return new TextDecoder().decode(decompressed)
//         } catch (error) {
//             // console.error('Decompression failed:', error)
//             // Return a fallback or partial decompression result if necessary
//             return ''
//         }
//     }

//     getLength() {
//         return this.vocab.size
//     }
// }

// class BinaryTokenizer {
//     constructor({
//         minLength = 1,
//         maxLength = 8,
//         corpus = '',
//         chunkSize = 1024 * 1024
//     } = {}) {
//         this.vocab = new Map() // Maps binary pattern to index
//         this.indexToPattern = new Map() // Maps index to binary pattern for decoding
//         this.minLength = minLength
//         this.maxLength = maxLength
//         this.corpus = corpus
//         this.chunkSize = chunkSize // Adjust based on your environment
//         this.train()
//     }

//     train() {
//         const encoder = new TextEncoder()
//         let index = 0 // Start indexing from 0
//         let mb = 0
//         for (
//             let start = 0;
//             start < this.corpus.length;
//             start += this.chunkSize
//         ) {
//             console.log(`chunking: ${mb}mb, length: ${this.getLength()}`)
//             mb++
//             const end = Math.min(start + this.chunkSize, this.corpus.length)
//             const chunk = this.corpus.substring(start, end)
//             const compressed = pako.deflate(encoder.encode(chunk))
//             let binaryString = ''
//             compressed.forEach((byte) => {
//                 binaryString += byte.toString(2).padStart(8, '0')
//             })

//             for (
//                 let length = this.minLength;
//                 length <= this.maxLength;
//                 length++
//             ) {
//                 for (let i = 0; i <= binaryString.length - length; i++) {
//                     const pattern = binaryString.substring(i, i + length)
//                     if (!this.vocab.has(pattern)) {
//                         this.vocab.set(pattern, index)
//                         this.indexToPattern.set(index, pattern)
//                         index++
//                     }
//                 }
//             }
//         }
//     }

// encode(string) {
//     const encoder = new TextEncoder()
//     const compressed = pako.deflate(encoder.encode(string))
//     let binaryString = ''
//     compressed.forEach((byte) => {
//         binaryString += byte.toString(2).padStart(8, '0')
//     })

//     const tokens = []
//     for (let i = 0; i < binaryString.length; ) {
//         let tokenFound = false
//         for (
//             let length = this.maxLength;
//             length >= this.minLength;
//             length--
//         ) {
//             if (i + length > binaryString.length) continue
//             const potentialToken = binaryString.substring(i, i + length)
//             if (this.vocab.has(potentialToken)) {
//                 tokens.push(this.vocab.get(potentialToken))
//                 i += length
//                 tokenFound = true
//                 break
//             }
//         }
//         if (!tokenFound) i++
//     }
//     return tokens
// }

// decode(tokens) {
//     let binaryString = ''
//     tokens.forEach((index) => {
//         const pattern = this.indexToPattern.get(index)
//         if (pattern) {
//             binaryString += pattern
//         } else {
//             // Ignore invalid tokens by not adding anything to binaryString
//         }
//     })

//     try {
//         const byteArray = []
//         for (let i = 0; i < binaryString.length; i += 8) {
//             byteArray.push(parseInt(binaryString.slice(i, i + 8), 2))
//         }
//         const decompressed = pako.inflate(new Uint8Array(byteArray))
//         return new TextDecoder().decode(decompressed)
//     } catch (error) {
//         // console.error('Decompression failed:', error)
//         // Return a fallback or partial decompression result if necessary
//         return ''
//     }
// }

//     getLength() {
//         return this.vocab.size
//     }

//     writeVocabularyToFile(path) {
//         // skip
//     }
// }

class CharacterTokenizer extends TokenizerBase {
    constructor() {
        super()
        this.padToken = '�'
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
            )
        )
        this.vocab.unshift(this.padToken)
    }

    getLength() {
        return this.vocab.length
    }

    encode(string) {
        return Array.from(string).map((char) => {
            const index = this.vocab.indexOf(char)
            return index !== -1 ? index : this.vocab.indexOf(this.padToken)
        })
    }

    decode(array) {
        return array
            .map((index) => {
                return index >= 0 && index < this.vocab.length
                    ? this.vocab[index]
                    : this.padToken
            })
            .join('')
    }
}

import { env } from '@xenova/transformers'
env.allowLocalModels = false
import { AutoTokenizer } from '@xenova/transformers'
class XenovaTokenizer {
    constructor(config) {
        this.model = config.model || 'openai-community/gpt2'
        this.tokenizer
    }

    async init() {
        this.tokenizer = await AutoTokenizer.from_pretrained(this.model)
    }

    getLength() {
        return this.tokenizer.model.vocab.length
    }

    encode(string) {
        return this.tokenizer.encode(string)
    }

    decode(array) {
        return this.tokenizer.decode(array, { skip_special_tokens: true })
    }

    writeVocabularyToFile(path) {
        // skip
    }
}

class BasicSubwordTokenizer {
    constructor(
        maxVocabSize = 32000,
        trainIterations = 50_000_000,
        corpus = shaks13
    ) {
        this.maxVocabSize = maxVocabSize
        const initialVocab =
            `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
                ''
            )
        this.vocab = new Map(initialVocab.map((token, index) => [token, index]))
        this.tokenFrequencies = new Map()
        this.train(corpus, trainIterations, 1, 7)
    }

    train(corpus, maxIterations = 10_000_000, minLength = 2, maxLength = 7) {
        console.log('training a tokenizer')

        let vocabSize = this.vocab.size
        let iterationCounter = 0

        while (
            vocabSize < this.maxVocabSize &&
            iterationCounter < maxIterations
        ) {
            let start = Math.floor(Math.random() * (corpus.length - maxLength))
            let len =
                minLength +
                Math.floor(Math.random() * (maxLength - minLength + 1))
            let token = corpus.substring(start, start + len)

            if (token && !this.vocab.has(token)) {
                this.tokenFrequencies.set(
                    token,
                    (this.tokenFrequencies.get(token) || 0) + 1
                )
            }

            iterationCounter++
            if (iterationCounter % 10_000_000 === 0) {
                console.log(
                    `Iteration ${iterationCounter}: Current vocab size is ${this.tokenFrequencies.size}`
                )
            }

            if (iterationCounter === maxIterations) {
                this.mergeVocabulary()
                if (this.vocab.size > vocabSize) {
                    vocabSize = this.vocab.size
                } else {
                    console.log(
                        `No new tokens found. Stopping early after ${iterationCounter} iterations.`
                    )
                    break
                }
            }
        }

        console.log(
            `Final vocab size: ${this.vocab.size} after ${iterationCounter} iterations.`
        )
    }

    mergeVocabulary() {
        const neededTokens = this.maxVocabSize - this.vocab.size
        const sortedTokens = Array.from(this.tokenFrequencies.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, neededTokens)

        sortedTokens.forEach(([token]) => {
            if (!this.vocab.has(token)) {
                this.vocab.set(token, this.vocab.size)
            }
        })
    }

    async writeVocabularyToFile(path = './data/models/ode') {
        if (typeof window === 'undefined') {
            const vocabJson = JSON.stringify(
                Array.from(this.vocab.keys()),
                null,
                2
            )
            const fs = await import('fs')
            fs.mkdirSync(path, { recursive: true })
            fs.writeFileSync(`${path}/tokenizer.json`, vocabJson, 'utf8')
            console.log(`Vocabulary written to ${path}`)
        }
    }

    getLength() {
        return this.vocab.size
    }

    decode(tokens) {
        return tokens
            .map((token) =>
                typeof token === 'number'
                    ? Array.from(this.vocab.keys())[token]
                    : token
            )
            .join('')
    }

    encode(text) {
        const tokens = []
        let currentToken = ''
        for (let i = 0; i < text.length; i++) {
            const char = text[i]
            const potentialToken = currentToken + char
            if (this.vocab.has(potentialToken)) {
                currentToken = potentialToken
            } else {
                if (currentToken) tokens.push(this.vocab.get(currentToken))
                currentToken = this.vocab.has(char) ? char : ''
            }
        }
        if (currentToken) tokens.push(this.vocab.get(currentToken))
        return tokens.filter((token) => token !== undefined)
    }
}

const tokenizers = {
    // CompressedBinaryTokenizer: (config) =>
    //     new CompressedBinaryTokenizer(config),
    CharacterTokenizer: (config) => new CharacterTokenizer(config),
    BasicSubwordTokenizer: (maxVocabSize, trainIterations, corpus) =>
        new BasicSubwordTokenizer(maxVocabSize, trainIterations, corpus),
    XenovaTokenizer: (config) => new XenovaTokenizer(config)
}
export default tokenizers

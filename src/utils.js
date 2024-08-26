export const colors = {
    BLUE: '\x1b[34m',
    GREEN: '\x1b[32m',
    RED: '\x1b[31m',
    WHITE: '\x1b[0m'
}

export const delay = (ms) => new Promise((res) => setTimeout(res, ms))

export function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[array[i], array[j]] = [array[j], array[i]]
    }
}

export function randomString(
    len = 3,
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
) {
    let text = ''
    for (let i = 0; i < len; i++) {
        text += chars.charAt(Math.floor(Math.random() * chars.length))
    }
    return text
}

export function randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min)
}

export class LinearCongruentialGenerator {
    constructor(seed) {
        this.seed = seed || Date.now()
        this.a = 1664525
        this.c = 1013904223
        this.m = Math.pow(2, 32)
    }

    nextInt() {
        this.seed = (this.a * this.seed + this.c) % this.m
        return this.seed
    }

    randomInt(min, max) {
        const range = max - min + 1
        return Math.floor(this.nextInt() / (this.m / range)) + min
    }

    randomFloat(min, max) {
        return (this.nextInt() / this.m) * (max - min) + min
    }

    pseudoRandomBetween(min, max) {
        return Math.floor(this.randomFloat(min, max))
    }
}

export function getRandomBiasedNumber(num1, num2, factor) {
    const min = Math.min(num1, num2)
    const max = Math.max(num1, num2)
    const power = factor
    const rnd = Math.random()
    const scaledRnd = Math.pow(rnd, power)
    const result = min + (max - min) * scaledRnd
    return Math.floor(result)
}

export function splitLines(text, num = 100) {
    const lines = text.split(/\r?\n/)
    return lines.slice(0, num).join('\n')
}

export function randomValueFromArray(array) {
    const randomIndex = Math.floor(Math.random() * array.length)
    return array[randomIndex]
}

export function generatePaddedNumbers(start, end, totalDigits) {
    const numbers = []
    for (let i = start; i <= end; i++) {
        numbers.push(String(i).padStart(totalDigits, '0'))
    }
    return numbers
}

// Return a deterministic value from any array
export function seededValueFromArray(array, seed) {
    return array[Math.floor(seededPRNG(seed) * array.length)]
}

// Generate a deterministic PRNG from a string
export function seededPRNG(str) {
    // Hash function
    function xmur3(str) {
        for (var i = 0, h = 1779033703 ^ str.length; i < str.length; i++) {
            h = Math.imul(h ^ str.charCodeAt(i), 3432918353)
            h = (h << 13) | (h >>> 19)
        }
        return function () {
            h = Math.imul(h ^ (h >>> 16), 2246822507)
            h = Math.imul(h ^ (h >>> 13), 3266489909)
            return (h ^= h >>> 16) >>> 0
        }
    }

    // 128 bit generator
    function xoshiro128ss(a, b, c, d) {
        return function () {
            var t = b << 9,
                r = a * 5
            r = ((r << 7) | (r >>> 25)) * 9
            c ^= a
            d ^= b
            b ^= c
            a ^= d
            c ^= t
            d = (d << 11) | (d >>> 21)
            return (r >>> 0) / 4294967296
        }
    }

    // Create xmur3 state:
    const state = xmur3(str.toString())
    // Output four 32-bit hashes to provide the seed for sfc32.
    const rand = xoshiro128ss(state(), state(), state(), state())
    // Obtain sequential random numbers like so:
    return rand()
}

export function* elapsedTimeGenerator() {
    let previousTime = new Date()

    while (true) {
        const currentTime = new Date()
        const elapsedTime = currentTime - previousTime
        previousTime = currentTime

        yield elapsedTime
    }
}

export function* emaGenerator(alpha = 0.01) {
    let ema = null
    while (true) {
        const newLoss = yield ema // Pause here and return exponential moving average
        if (newLoss !== undefined) {
            ema = ema === null ? newLoss : alpha * newLoss + (1 - alpha) * ema // Update EMA with the new loss value
        }
    }
}

export function findMatches(sequence1, sequence2) {
    let matchLength = 0

    // Find the length of the matching part
    for (let i = 0; i < Math.min(sequence1.length, sequence2.length); i++) {
        if (sequence1[i] === sequence2[i]) {
            matchLength++
        } else {
            break
        }
    }

    // Split the second sequence into matching and differing parts
    const matchPart = sequence2.substring(0, matchLength)
    const differPart = sequence2.substring(matchLength)

    return { old: matchPart, new: differPart }
}

export function preprocessData(
    text,
    tokenizer,
    maxSequenceLength,
    paddingSide = 'left'
) {
    let indices = text
    if (!Array.isArray(text)) {
        indices = tokenizer.encode(text)
    }

    // Ensure sequence is not longer than inputLength
    if (indices.length > maxSequenceLength) {
        indices = indices.slice(0, maxSequenceLength)
    }

    const padding = new Array(maxSequenceLength - indices.length).fill(0)
    if (paddingSide === 'left') {
        return padding.concat(indices)
    } else if (paddingSide === 'right') {
        return indices.concat(padding)
    } else if (paddingSide === 'random') {
        while (indices.length < maxSequenceLength) {
            if (Math.random() < 0.5) {
                indices.push(0)
            } else {
                indices.unshift(0)
            }
        }
        return indices
    } else {
        return indices
    }
}

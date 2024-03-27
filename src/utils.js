import { shaks13 } from './data.js'

export const colors = {
    BLUE: '\x1b[34m',
    GREEN: '\x1b[32m',
    RED: '\x1b[31m',
    WHITE: '\x1b[0m'
}

export function randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min)
}

export function splitLines(text, num = 100) {
    const lines = text.split(/\r?\n/)
    return lines.slice(0, num).join('\n')
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
    const state = xmur3(str)
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
    let indices = tokenizer.encode(text)

    // Ensure sequence is not longer than inputLength
    if (indices.length > maxSequenceLength) {
        indices = indices.slice(0, maxSequenceLength)
    }

    const padding = new Array(maxSequenceLength - indices.length).fill(0)
    if (paddingSide === 'left') {
        return padding.concat(indices)
    } else if (paddingSide === 'right') {
        return indices.concat(padding)
    } else {
        while (indices.length < maxSequenceLength) {
            if (Math.random() < 0.5) {
                indices.push(0)
            } else {
                indices.unshift(0)
            }
        }
        return indices
    }
}

function* stringSampler(sampleLen, overfit = 0, str = shaks13) {
    if (overfit > 0) {
        str = splitLines(str, overfit)
    }
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

function* sequentialStringSampler(sampleLen, str) {
    let index = 0
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index++ // Make 1 time step over the str
    }
}

import fs from 'fs'
import path from 'path'

function directorySampler(
    sampleLen,
    overfit = 0,
    dir = './',
    delimiter = '\n\n'
) {
    let allText = ''

    const readDirSync = (currentPath) => {
        fs.readdirSync(currentPath, { withFileTypes: true }).forEach(
            (entry) => {
                const entryPath = path.join(currentPath, entry.name)
                if (entry.isDirectory()) {
                    readDirSync(entryPath)
                } else {
                    allText += `${fs.readFileSync(entryPath, 'utf8')}${delimiter}`
                }
            }
        )
    }

    readDirSync(dir)

    return stringSampler(sampleLen, overfit, allText)
}

export const samplers = {
    stringSampler: (sampleLen, overfit, str) =>
        stringSampler(sampleLen, overfit, str),
    sequentialStringSampler: (sampleLen, str) =>
        sequentialStringSampler(sampleLen, str),
    directorySampler: (sampleLen, overfit, dir, delimiter) =>
        directorySampler(sampleLen, overfit, dir, delimiter)
}

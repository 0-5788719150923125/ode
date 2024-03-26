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

import { delay } from './utils.js'

function* stringSampler(sampleLen, overfit = 0, str = '') {
    if (overfit > 0) str = splitLines(str, overfit)
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

function* sinWaveOscillator(min, max) {
    let i = 0
    const amplitude = (max - min) / 2 // Height from center to peak
    const offset = (max + min) / 2 // Midpoint between max and min
    while (true) {
        const value = Math.sin(i) * amplitude + offset
        yield Math.round(value)
        i += 0.1 // Increment angle to move along the sine wave
    }
}

function* sequentialStringSampler(sampleLen, overfit = 0, str) {
    let index = 0
    const oscillator = sinWaveOscillator(1, 64)
    if (overfit > 0) str = splitLines(str, overfit)
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index = index + oscillator.next().value // Make [x] time step over the str
    }
}

// async function directorySampler(
//     sampleLen,
//     overfit = 0,
//     dirs = ['./'],
//     delimiter = '\n\n'
// ) {
//     let allText = await directoryReader(dirs, delimiter)
//     return stringSampler(sampleLen, overfit, allText)
// }

async function directorySampler(dirs = './', delimiter = '\n\n') {
    const fs = (await import('fs')).default
    const path = (await import('path')).default

    let allText = ''

    const readDirSync = (dir) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true })
        for (const entry of entries) {
            const entryPath = path.join(dir, entry.name)
            if (entry.isDirectory()) {
                readDirSync(entryPath)
            } else {
                try {
                    const fileContent = fs.readFileSync(entryPath, 'utf8')
                    if (fileContent.trim() !== '') {
                        allText += `${fileContent}${delimiter}`
                    }
                } catch (error) {
                    if (error.code === 'ENOENT') {
                        // console.warn(`File not found: ${entryPath}`)
                    } else if (error.message.includes('Invalid UTF-8 data')) {
                        // console.warn(
                        //     `Skipping file with invalid UTF-8 encoding: ${entryPath}`
                        // )
                    } else {
                        // console.warn(`Error reading file: ${entryPath}`, error)
                    }
                }
            }
        }
    }

    const directories = dirs.split(',')

    for (const directory of directories) {
        readDirSync(directory)
    }

    return allText
}

async function fetchURLSampler(
    url = 'https://www.gutenberg.org/files/100/old/shaks12.txt',
    path = 'shaks12.txt'
) {
    const fs = await import('fs')

    async function fetchAndSaveContent(url, filePath) {
        const response = await fetch(url)
        const text = await response.text()

        fs.mkdirSync('./data/datasets', { recursive: true })
        fs.writeFile(filePath, text, (err) => {
            if (err) {
                console.error('Error saving file:', err)
            } else {
                console.log('File saved successfully!')
            }
        })
        return text
    }
    const text = await fetchAndSaveContent(url, `./data/datasets/${path}`)
    console.log(text.slice(0, 300))
    return text
}

const samplers = {
    stringSampler: (sampleLen, overfit, str) =>
        stringSampler(sampleLen, overfit, str),
    sequentialStringSampler: (sampleLen, overfit, str) =>
        sequentialStringSampler(sampleLen, overfit, str),
    directorySampler: (dir, delimiter) => directorySampler(dir, delimiter),
    fetchURLSampler: (url, path) => fetchURLSampler(url, path)
}
export default samplers

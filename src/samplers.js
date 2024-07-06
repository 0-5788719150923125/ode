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

function* sequentialStringSampler(sampleLen, str) {
    let index = Math.floor(Math.random() * (str.length - sampleLen + 1))
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index++
    }
}

async function directorySampler(dirs = './', delimiter = '\n\n') {
    const fs = (await import('fs')).default
    const path = (await import('path')).default

    let allText = ''

    const isValidUtf8 = (buffer) => {
        return buffer.every((byte) => byte <= 127)
    }

    const readDirSync = (dir) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true })
        for (const entry of entries) {
            const entryPath = path.join(dir, entry.name)
            if (entry.isDirectory()) {
                readDirSync(entryPath)
            } else {
                const fileBuffer = fs.readFileSync(entryPath)
                if (isValidUtf8(fileBuffer)) {
                    const fileContent = fileBuffer.toString('utf8')
                    if (fileContent.trim() !== '') {
                        allText += `${fileContent}${delimiter}`
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

async function* CosmopediaSampler(sampleLen) {
    const CosmopediaDataset = (await import('./samplers/cosmopedia.js')).default
    const sampler = new CosmopediaDataset()
    await sampler.init()
    sampler.loadSchema([{ prompt: 'INPUT: ' }, { text: 'OUTPUT: ' }])
    while (true) {
        yield await sampler.getSample(sampleLen)
    }
}

const samplers = {
    stringSampler: (sampleLen, overfit, str) =>
        stringSampler(sampleLen, overfit, str),
    sequentialStringSampler: (sampleLen, str) =>
        sequentialStringSampler(sampleLen, str),
    directorySampler: (dir, delimiter) => directorySampler(dir, delimiter),
    fetchURLSampler: (url, path) => fetchURLSampler(url, path),
    CosmopediaSampler: (sampleLen) => CosmopediaSampler(sampleLen)
}
export default samplers

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

async function directorySampler(
    sampleLen,
    overfit = 0,
    dir = './',
    delimiter = '\n\n'
) {
    const fs = (await import('fs')).default
    const path = (await import('path')).default

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

async function gunSampler() {
    const Gun = (await import('gun')).default
    // const SEA = (await import('gun/sea.js')).default

    const gun = new Gun({
        peers: ['wss://59.src.eco/gun', 'wss://95.src.eco/gun'],
        localStorage: true,
        radisk: false,
        axe: false,
        file: './data/gun'
    })

    gun.get('src')
        .get('bullets')
        .get('trade')
        .on(async (node) => {
            console.log(node)
        })
}

const samplers = {
    stringSampler: (sampleLen, overfit, str) =>
        stringSampler(sampleLen, overfit, str),
    sequentialStringSampler: (sampleLen, str) =>
        sequentialStringSampler(sampleLen, str),
    directorySampler: (sampleLen, overfit, dir, delimiter) =>
        directorySampler(sampleLen, overfit, dir, delimiter),
    gunSampler: () => gunSampler()
}
export default samplers

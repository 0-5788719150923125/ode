import { createCanvas } from 'canvas'
import fs from 'fs'

function createTextImage(text) {
    const size = 500
    const backgroundColor = 'white'
    const textColor = 'black'
    const fontFamily = 'Arial'
    const padding = 20

    const canvas = createCanvas(size, size)
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, size, size)

    ctx.fillStyle = textColor
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'

    let fontSize = 32
    let lineHeight = fontSize * 1.2
    let paragraphs = text.split('\n')
    let lines = []

    while (fontSize > 4) {
        ctx.font = `${fontSize}px ${fontFamily}`
        lines = []

        for (let i = 0; i < paragraphs.length; i++) {
            let words = paragraphs[i].split(' ')
            let currentLine = ''

            for (let j = 0; j < words.length; j++) {
                let word = words[j]
                let width = ctx.measureText(`${currentLine} ${word}`).width

                if (width < size - padding * 2) {
                    currentLine += `${word} `
                } else {
                    lines.push(currentLine.trim())
                    currentLine = `${word} `
                }
            }

            if (currentLine !== '') {
                lines.push(currentLine.trim())
            }
        }

        lineHeight = fontSize * 1.2
        let totalHeight = lineHeight * lines.length

        if (totalHeight <= size - padding * 2) {
            break
        }

        fontSize -= 1
    }

    let y = (size - lineHeight * lines.length) / 2

    for (const line of lines) {
        if (line === '') {
            y += lineHeight // Skip empty lines between paragraphs
        } else {
            ctx.fillText(line, padding, y)
            y += lineHeight
        }
    }

    const buffer = canvas.toBuffer('image/png')
    fs.writeFileSync('output.png', buffer)

    return getPixels(ctx, size)
}

function getPixels(ctx, size) {
    const imageData = ctx.getImageData(0, 0, size, size)
    const pixels = imageData.data
    const tokenizedData = []

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i]
        const g = pixels[i + 1]
        const b = pixels[i + 2]
        const a = pixels[i + 3]

        // Check if the pixel is text (black) or whitespace (white)
        const isText = r === 0 && g === 0 && b === 0 && a === 255
        const token = isText ? 1 : 0
        tokenizedData.push(token)
    }
    return tokenizedData
}

// Example usage
const textData = ` In the small German village of Röcken bei Lützen, located in a rural farmland area about 20 miles southwest of Leipzig, Friedrich Wilhelm Nietzsche was born at approximately 10:00 a.m. on October 15, 1844. The date coincided with the 49th birthday of the Prussian King, Friedrich Wilhelm IV, after whom Nietzsche was named, and who had been responsible for Nietzsche’s father’s appointment as Röcken’s town pastor.`
createTextImage(textData)

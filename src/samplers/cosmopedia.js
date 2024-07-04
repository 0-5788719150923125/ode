import * as arrow from 'apache-arrow'
import wasmInit, { readParquet, Table } from 'parquet-wasm'
import {
    generatePaddedNumbers,
    randomBetween,
    randomValueFromArray,
    shuffleArray
} from '../utils.js'

if (typeof window !== 'undefined') await wasmInit()

export default class CosmopediaDataset {
    constructor(config) {
        this.dataset = 'HuggingFaceTB/cosmopedia'
        this.slice = 'stories'
        this.split = 'train'
        this.shards = generatePaddedNumbers(0, 43, 5)
        this.delimiter = '\n\n'
        this.cacheSize = 20000
        this.cachedText = ''
    }

    async init() {
        await this.fetchShard()
        this.moveDataIntoTable()
    }

    async fetchShard() {
        const shard = randomValueFromArray(this.shards)
        const path = `data/${this.slice}/${
            this.split
        }-${shard}-of-${this.shards.slice(-1)}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(url)
        const response = await fetch(url)

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        this.buffer = new Uint8Array(await response.arrayBuffer())
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        const arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())
    }

    viewSchema() {
        console.log(table.schema.toString())
    }

    loadSchema(array) {
        this.schema = []
        array.map((obj) => {
            Object.entries(obj).forEach(([key, value]) => {
                const idx = this.table.schema.fields.findIndex(
                    (field) => field.name === key
                )
                console.assert(
                    idx !== -1,
                    `the key of ${key} does not exist in this dataset`
                )
                this.schema.push({ idx, value })
            })
        })
    }

    fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            const batchIdx = randomBetween(0, this.table.batches.length)

            const text = []

            let rowIdx
            for (const obj of this.schema) {
                const column = this.table.batches[batchIdx].getChildAt(obj.idx)
                if (!rowIdx) {
                    rowIdx = randomBetween(0, column.length)
                }
                const prefix = obj.value
                text.push(prefix + column.get(rowIdx))
            }

            this.cachedText += text.join(this.delimiter) + this.delimiter
        }
    }

    async getSample(size = 512) {
        this.fillCache()
        const sample = this.cachedText.slice(0, size)
        this.cachedText = this.cachedText.slice(size, -1)
        return sample
    }
}

async function main() {
    const sampler = new CosmopediaDataset()
    await sampler.init()
    sampler.loadSchema([{ prompt: 'PROMPT: ' }, { text: 'ASSISTANT: ' }])
    for (let i = 0; i < 10; i++) {
        console.log(sampler.getSample())
        console.log('---')
        console.log('---')
        console.log('---')
    }
}

main()

import { Table } from 'apache-arrow'
import { parseRecordBatch } from 'arrow-js-ffi'
import initWasm, { wasmMemory, readParquetStream } from 'parquet-wasm'
import { randomBetween, randomValueFromArray } from '../utils.js'

export default class FinewebDataset {
    constructor(config) {
        this.dataset = 'HuggingFaceFW/fineweb'
        this.slices = [{ slice: 'CC-MAIN-2024-18', shards: 50, chunks: 5 }]
        this.split = 'train'
        this.delimiter = '\n\n'
        this.eosToken = config?.eosToken || '֍'
        this.batchesBeforeRefresh = config?.batchesBeforeRefresh || 10000
        this.batches = 0
        this.cacheSize = 20000
        this.cachedText = ''
        this.table = {}
        this.schemaTemplate = config?.schema
    }

    async init() {
        const isBrowser =
            (typeof self !== 'undefined' &&
                typeof self.importScripts === 'function') ||
            typeof window !== 'undefined'
        if (isBrowser) await initWasm()
        await this.fetchRandomShard()
    }

    generatePaddedNumbers(shards, chunks) {
        const numbers = []
        for (let i = 0; i <= shards; i++) {
            for (let j = 0; j <= chunks; j++) {
                const prefix = String(j).padStart(3, '0')
                const suffix = String(i).padStart(5, '0')
                numbers.push(`${prefix}_${suffix}`)
            }
        }
        return numbers
    }

    async fetchRandomShard() {
        const { slice, shards, chunks } = randomValueFromArray(this.slices)
        const shardIndices = this.generatePaddedNumbers(shards, chunks)
        const shard = randomValueFromArray(shardIndices)
        const path = `data/${slice}/${shard}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log('fetching shard:', `${shard}`, 'slice:', slice)
        try {
            await this.streamDataIntoTable(url)
        } catch (err) {
            console.error(err)
            console.warn(
                `Failed to fetch shard (${shard}) from HuggingFace! We will continue using the old one for now...`
            )
        }
        this.loadSchema(this.schemaTemplate)
    }

    async streamDataIntoTable(url) {
        const stream = await readParquetStream(url)
        // Read Parquet buffer to Arrow Table
        const batches = []

        const chance = 0.1
        for await (const wasmRecordBatch of stream) {
            // Used to keep memory usage down (original Parquet files are > 2GB in size)
            if (Math.random() > chance) continue
            console.log(Math.random())
            const ffiRecordBatch = wasmRecordBatch.intoFFI()
            const recordBatch = parseRecordBatch(
                wasmMemory().buffer,
                ffiRecordBatch.arrayAddr(),
                ffiRecordBatch.schemaAddr()
            )
            batches.push(recordBatch)
        }
        // Convert to JS Arrow Table
        this.table = new Table(batches)
    }

    loadSchema(array = [{ text: 'INPUT: ' }]) {
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

    async fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            let batchIdx = randomBetween(0, this.table.batches.length - 1)

            const text = []

            let rowIdx = null
            for (const field of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(field.idx)
                if (rowIdx === null) {
                    rowIdx = randomBetween(0, column.length - 1)
                }
                const prefix = field.value
                const data = column.get(rowIdx)
                text.push(prefix + data)
            }
            this.cachedText += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample(size = 512) {
        this.batches++
        try {
            if (this.batches % this.batchesBeforeRefresh === 0) {
                await this.fetchRandomShard()
            }
            await this.fillCache()
            const sample = this.cachedText.slice(0, size)
            this.cachedText = this.cachedText.slice(size)
            return sample
        } catch (err) {
            console.error(err)
            return await this.getSample(size)
        }
    }
}

async function main() {
    const sampler = new FinewebDataset({
        schema: [{ text: '' }]
    })
    await sampler.init()
    for (let i = 0; i < 10; i++) {
        console.log(await sampler.getSample())
        console.log('---')
        console.log('---')
        console.log('---')
    }
}

main()

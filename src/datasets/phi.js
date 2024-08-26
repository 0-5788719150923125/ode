import ParquetReader from './readers/parquet.js'
import { LinearCongruentialGenerator } from '../utils.js'

export default class PhiDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'open-phi/textbooks'
        this.schemaTemplate = config?.schema || [{ markdown: '\n\n' }]
        this.seed = config?.seed || 42
        this.rng = new LinearCongruentialGenerator(this.seed)
    }

    async fetchRandomShard() {
        // There's only 1 shard, no need to keep reloading them here.
        if (this.loaded) return
        const shard = 'train-00000-of-00001-b513d9e388d56453.parquet'
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/data/${shard}`
        console.log('fetching dataset:', this.dataset, 'shard:', `${shard}`)
        try {
            await this.streamDataIntoTable(url)
        } catch (err) {
            console.error(err)
            console.warn(
                `Failed to fetch shard (${shard}) from HuggingFace! We will continue using the old one for now...`
            )
        }
        this.loadSchema(this.schemaTemplate)
        this.loaded = true
    }

    async fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            let batchIdx = this.rng.pseudoRandomBetween(
                0,
                this.table.batches.length - 1
            )

            const text = []

            let rowIdx = null
            for (const field of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(field.idx)
                if (rowIdx === null) {
                    rowIdx = this.rng.pseudoRandomBetween(0, column.length - 1)
                    // console.log(
                    //     `has ${this.table.batches.length} batches, with ${
                    //         column.length
                    //     } rows, and ${
                    //         column.length * this.table.batches.length
                    //     } est combinations`
                    // )
                }
                const prefix = field.value
                const data = column.get(rowIdx)
                text.push(prefix + data)
            }
            this.cachedText += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample({ mode = 'train', size = 512 }) {
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
            return await this.getSample({ mode, size })
        }
    }
}

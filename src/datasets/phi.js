import ParquetReader from './readers/parquet.js'
import { randomBetween } from '../utils.js'

export default class PhiDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'open-phi/textbooks'
        this.schemaTemplate = config?.schema || [{ markdown: '\n\n' }]
    }

    async fetchRandomShard() {
        // There's only 1 shard, no need to keep reloading them here.
        if (this.loaded) return
        const shard = 'train-00000-of-00001-b513d9e388d56453.parquet'
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/data/${shard}`
        console.log('fetching dataset:', this.dataset, 'shard:', `${shard}`)
        await this.streamDataIntoTable(url)
        this.loadSchema(this.schemaTemplate)
        this.loaded = true
    }

    async fillCache(mode = 'train', batchIdx = 0) {
        while (this.cachedText[mode].length < this.cacheSize) {
            const text = []

            let rowIdx = null
            for (const field of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(field.idx)
                if (rowIdx === null) {
                    rowIdx = this.rng[mode].randomBetween(0, column.length - 1)
                }
                const prefix = field.value
                const data = column.get(rowIdx)
                text.push(prefix + data)
            }
            this.cachedText[mode] += text.join(this.delimiter) + this.eosToken
        }
    }

    async take({ mode = 'train', size = 512 }) {
        let batchIdx = this.trainBatchIdx
        if (this.seed === null) batchIdx = randomBetween(0, 1)
        if (mode === 'validation') batchIdx = this.validationBatchIdx
        await this.fillCache(mode, batchIdx)
        const sample = this.cachedText[mode].slice(0, size)
        this.cachedText[mode] = this.cachedText[mode].slice(size)
        return sample
    }
}

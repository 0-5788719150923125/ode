import ParquetReader from './readers/parquet.js'
import { LinearCongruentialGenerator, randomBetween } from '../utils.js'

export default class PhiDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'open-phi/textbooks'
        this.schemaTemplate = config?.schema || [{ markdown: '\n\n' }]
        this.seed = config?.seed || null
        if (this.seed !== null) {
            console.log(`phi dataset had a seed, using it: (${this.seed})`)
            this.lcg = new LinearCongruentialGenerator(this.seed)
        }

        this.trainBatchIdx = 0
        this.validationBatchIdx = 1
        this.cachedText = {
            train: '',
            validation: ''
        }
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

    async fillCache(mode, batchIdx = 0) {
        while (this.cachedText[mode].length < this.cacheSize) {
            const text = []

            let rowIdx = null
            for (const field of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(field.idx)
                if (rowIdx === null) {
                    if (this.seed) {
                        rowIdx = this.lcg.pseudoRandomBetween(
                            0,
                            column.length - 1
                        )
                    } else {
                        rowIdx = randomBetween()
                    }
                }
                const prefix = field.value
                const data = column.get(rowIdx)
                text.push(prefix + data)
            }
            this.cachedText[mode] += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample({ mode = 'train', size = 512 }) {
        let batchIdx = this.trainBatchIdx
        if (this.seed === null) batchIdx = randomBetween(0, 1)
        if (mode === 'validation') batchIdx = this.validationBatchIdx
        await this.fillCache(mode, batchIdx)
        const sample = this.cachedText[mode].slice(0, size)
        this.cachedText[mode] = this.cachedText[mode].slice(size)
        return sample
    }
}

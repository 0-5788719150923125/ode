import ParquetReader from './readers/parquet.js'
import { randomValueFromArray } from '../utils.js'

export default class WikipediaDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'wikimedia/wikipedia'
        this.slices = [{ slice: '20231101.en', shards: 41 }]
        this.schemaTemplate = config?.schema || [
            { title: 'INPUT: ', text: 'OUTPUT: ' }
        ]
    }

    generatePaddedNumbers(split, numShards) {
        const numbers = []
        const suffix = String(numShards).padStart(5, '0')
        for (let i = 0; i < numShards; i++) {
            const prefix = String(i).padStart(5, '0')
            numbers.push(`${split}-${prefix}-of-${suffix}`)
        }
        return numbers
    }

    async fetchRandomShard() {
        const { slice, shards } = randomValueFromArray(this.slices)
        const shardIndices = this.generatePaddedNumbers(this.split, shards)
        const shard = randomValueFromArray(shardIndices)
        const path = `${slice}/${shard}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(
            'fetching dataset:',
            this.dataset,
            'shard:',
            `${shard}`,
            'slice:',
            slice
        )
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
}

import ParquetReader from './readers/parquet.js'

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
}

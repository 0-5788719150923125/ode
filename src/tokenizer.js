import { Llama2Tokenizer } from '@lenml/llama2-tokenizer'
import { load_vocab } from '@lenml/llama2-tokenizer-vocab-llama2'

export default class Tokenizer {
    constructor() {
        // this.padToken = '�'
        // this.vocab = Array.from(
        //     new Set(
        //         `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
        //     )
        // )
        // this.vocab.unshift(this.padToken)
        this.model = new Llama2Tokenizer()
        this.model.install_vocab(load_vocab())
    }

    getLength() {
        return this.model.vocab_size
    }
}

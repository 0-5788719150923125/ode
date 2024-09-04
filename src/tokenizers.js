import TokenMonsterCore from './tokenizers/tokenmonster.cjs'
import { env } from '@xenova/transformers'
env.allowLocalModels = typeof window !== 'undefined' ? true : false
import { AutoTokenizer } from '@xenova/transformers'

class TokenizerBase {
    constructor() {
        // pass
    }

    getLength() {
        return this.vocab.length
    }

    getConfig() {
        return {
            class: this.constructor.name
        }
    }

    encode(string) {
        // not implemented
    }

    decode(array) {
        // not implemented
    }

    writeVocabularyToFile(path) {
        // skip
    }
}

class CharacterTokenizer extends TokenizerBase {
    constructor(config) {
        super(config)
        const vocab =
            config?.vocab ||
            `\n \r\n \r \u2028\u2029!"#$%&'()*+,-.\\/0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_\`abcdefghijklmnopqrstuvwxyz{|}~¡£§©¬®°±²³µ¶·¹º»¼½ÀÁÄÅÇÉÎÓÔÖ×ØÜßàáâãäåæçèéêëíîïðñóôõö÷øúüĀāăćČčđēěğħīİıłńōőśşšūůŷźŽžȧȲȳɛʌʻʼˆ˚̸̂̄̅ΑΒΓΔΘΛΠΣΦΨΩέήίαβγδεζηθικλμνοπρςστφχψωόϵкѰḗṓợἴὁ​‐–—‘’“”†•․… ′″⁴⁻₀₁₂₃₄ₖₙ€ℓℕℚℝ™ℤ⅓←↑→↓↔↦⇌⇒⇔⇠⇥∀∂∃∅∆∇∈∉∏∑−∗∘∙√∞∠∧∨∩∪∫∼≅≈≠≡≤≥≫⊂⊆⊕⊙⋅⌈⌉⌘─│└├■□▶◆○●◦♢♥♦✓❤⟨⟩⨯⩽ⱼⲜ。・世前务发后告周在将我报新更末本界的给请财送道\ud835\ud835𝝅\udf48\udf73`
        this.padToken = '�'
        this.tokens = Array.from(new Set(vocab))
        this.tokens.unshift(this.padToken)
        console.log('Parsed vocabulary:')
        console.log(JSON.stringify(this.tokens.sort()))
    }

    getLength() {
        return this.tokens.length
    }

    getConfig() {
        return {
            ...super.getConfig(),
            length: this.getLength()
        }
    }

    encode(string) {
        return Array.from(string).map((char) => {
            const index = this.tokens.indexOf(char)
            return index !== -1 ? index : this.tokens.indexOf(this.padToken)
        })
    }

    decode(array) {
        return array
            .map((index) => {
                return index >= 0 && index < this.tokens.length
                    ? this.tokens[index]
                    : this.padToken
            })
            .join('')
    }
}

class XenovaTokenizer extends TokenizerBase {
    constructor(config) {
        super(config)
        this.model = config.model || 'openai-community/gpt2'
        this.tokenizer
    }

    async init() {
        this.tokenizer = await AutoTokenizer.from_pretrained(this.model)
    }

    getLength() {
        return this.tokenizer.model.vocab.length
    }

    getConfig() {
        return {
            ...super.getConfig(),
            model: this.model,
            length: this.getLength()
        }
    }

    encode(string) {
        return this.tokenizer.encode(string)
    }

    decode(array) {
        return this.tokenizer.decode(array, { skip_special_tokens: true })
    }
}

// https://github.com/alasdairforsythe/tokenmonster
class TokenMonster extends TokenizerBase {
    constructor(config) {
        super(config)
        this.model = config.model || 'englishcode-32000-consistent-v1'
    }

    async init() {
        this.tokenizer = new TokenMonsterCore()
        await this.tokenizer.load(
            `https://huggingface.co/alasdairforsythe/tokenmonster/raw/main/vocabs/${this.model}.vocab`
        )
        this.decoder = this.tokenizer.Decoder()
    }

    getLength() {
        return this.tokenizer.vocab_size
    }

    getConfig() {
        return {
            ...super.getConfig(),
            model: this.model,
            length: this.getLength()
        }
    }

    encode(string) {
        return this.tokenizer.tokenize(string)
    }

    decode(array) {
        return this.decoder.detokenize(array)
    }
}

const tokenizers = {
    CharacterTokenizer: (config) => new CharacterTokenizer(config),
    XenovaTokenizer: (config) => new XenovaTokenizer(config),
    TokenMonster: (config) => new TokenMonster(config)
}
export default tokenizers

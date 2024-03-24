from transformers import pipeline, LlamaTokenizer, AutoTokenizer
import torch


class LocalGenModel():
    def __init__(self, name):
        self.name = name
        # use your own model dir
        if self.name == 'alpaca-7B':
            self.dir = "path to your alpaca-7B model"
            self.tokenizer = LlamaTokenizer.from_pretrained(self.dir)
        elif self.name == 'gptneox-20B':
            self.dir = "path to your gptneox-20B model"
            self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        else:
            raise NotImplementedError
        self.eos_token_id = self.tokenizer.eos_token_id
        self.gen = pipeline(task='text-generation', model=self.dir,
                            torch_dtype=torch.float16, device_map="auto",)
        banwords = ['He', 'His', 'She', 'Her', 'It', 'Its', 'They', 'Their', 'This', 'These']
        banwordss = []
        for w in banwords:
            banwordss.append(w)
            banwordss.append(w.lower())
            banwordss.append(" " + w)
            banwordss.append(" " + w.lower())
        self.ban_pronoun_ids = self.tokenizer(banwordss, add_special_tokens=False).input_ids

    def get_output(self, input_text, max_tokens, temperature=1.0, ban_pronoun=False,):

        paradict = {'text_inputs': input_text,
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'pad_token_id': self.eos_token_id,
                    }
        if temperature == 0:
            paradict['do_sample'] = False
        if ban_pronoun:
            paradict['bad_words_ids'] = self.ban_pronoun_ids
        output_text = self.gen(**paradict)[0]['generated_text']
        return output_text[len(input_text):]

import copy
import openai
import time



def get_from_file(fn):
    with open(fn) as f:
        x = f.read()
    return x


class api_handler:
    def __init__(self, model, api_file):
        openai.api_key = get_from_file(api_file)
        self.model = model
        self.interval = 3 if self.model in ['codex'] else 1
        self.interval = 0 if 'edit' in self.model else self.interval
        if self.model == 'instructgpt':
            self.engine = 'text-davinci-002'
        elif self.model == 'newinstructgpt':
            self.engine = 'text-davinci-003'
        elif self.model == 'oldinstructgpt':
            self.engine = 'text-davinci-001'
        elif self.model == 'gpt3':
            self.engine = 'davinci'
        elif self.model == 'codex':
            self.engine = 'code-davinci-002'
        elif self.model == 'gpt3-edit':
            self.engine = 'text-davinci-edit-001'
        elif self.model == 'codex-edit':
            self.engine = 'code-davinci-edit-001'
        else:
            raise NotImplementedError

    def get_output(self, input_text, max_tokens, temperature=0,
                   suffix=None, stop=None, do_tunc=False, echo=False, ban_pronoun=False,
                   frequency_penalty=0, presence_penalty=0, return_prob=True):
        if self.model == 'codex':
            time.sleep(self.interval)
        try:
            if 'edit' in self.model:
                response = openai.Edit.create(
                    model=self.engine,
                    input="",
                    instruction=input_text,
                    temperature=temperature,
                    top_p=1,
                )
            elif ban_pronoun:
                # ban words like he, she, their to ensure question is clear without context
                ban_token_ids = [1119, 1375, 679, 632, 770, 2312, 5334, 2332,
                                 2399, 6363, 484, 673, 339, 340, 428, 777, 511,
                                 607, 465, 663, 2990, 3347, 1544, 1026,  1212,
                                 4711, 14574, 9360, 6653, 20459, 9930, 7091, 258,
                                 270,  5661, 27218, 24571, 372, 14363, 896,
                                 # 464,1169,383,262
                                 ]
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=input_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    suffix=suffix,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    echo=echo,
                    logit_bias={str(tid):-100 for tid in ban_token_ids}
                )
            else:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=input_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    suffix=suffix,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    echo=echo,
                    logprobs=1,
                )

            x = response['choices'][0]['text']

            if do_tunc or ('edit' in self.model):
                y = x.strip()
                if '\n' in y:
                    pos = y.find('\n')
                    y = y[:pos]
                if 'Q:' in y:
                    pos = y.find('Q:')
                    y = y[:pos]
                if 'Question:' in y:
                    pos = y.find('Question:')
                    y = y[:pos]
                assert not ('\n' in y)
                if not return_prob:
                    return y

            if not return_prob:
                return x

            if 'edit' not in self.model:
                output_token_offset = [i - len(input_text) for i in response['choices'][0]['logprobs']['text_offset']]
                output_token_tokens = response['choices'][0]['logprobs']['tokens']
                output_token_probs = response['choices'][0]['logprobs']['token_logprobs']
                output_token_offset_real = []
                output_token_tokens_real = []
                output_token_probs_real = []
                dx = copy.deepcopy(x)
                for idx, (offset, token, prob) in enumerate(
                        zip(output_token_offset, output_token_tokens, output_token_probs)):
                    if idx == 0 and token == ':': continue
                    if idx == 0 and not dx.startswith(token) and (" " + dx).startswith(token):
                        dx = " " + dx
                    if dx.startswith(token):
                        output_token_offset_real.append(offset)
                        output_token_tokens_real.append(token)
                        output_token_probs_real.append(prob)
                        dx = dx[len(token):]
                    else:
                        break
            else:
                output_token_offset_real, output_token_tokens_real, output_token_probs_real = [], [], []
            return x, (output_token_offset_real, output_token_tokens_real, output_token_probs_real)
        except Exception as e:
            if 'You exceeded your current quota, please check your plan and billing details.' in str(e):
                print("Exit because no quota")
                exit()
            time.sleep(2 * self.interval)
            return self.get_output(input_text, max_tokens, temperature=temperature,
                   suffix=suffix, stop=stop, do_tunc=do_tunc, echo=echo, ban_pronoun=ban_pronoun,
                   frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, return_prob=return_prob)

#!/usr/bin/env python3
"""Freeze additional checks and speed prompts with the verified model tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/deepseek'))
from encoding import encode_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    control = json.loads((ROOT / 'results/20261017T210718Z-iq2-streaming/fixture.json').read_text())
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    for case in control['checks'] + control['speed']:
        assert encode(case['rendered_prompt']) == case['input_ids'], case['id']
    args.output.mkdir(parents=True, exist_ok=True)

    def case(name, content, expected=None):
        messages = [{'role': 'user', 'content': content}]
        rendered = encode_messages(messages, thinking_mode='chat')
        ids = encode(rendered)
        result = {'id': name, 'messages': messages, 'rendered_prompt': rendered,
                  'input_ids': ids, 'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        if expected is not None:
            result['expected'] = expected
        return result

    checks = [
        ('multiply2', 'Compute 483 * 27 - 119. Return only the integer.', str(483 * 27 - 119)),
        ('modulo', 'What is 7**13 modulo 19? Return only the integer.', str(pow(7, 13, 19))),
        ('hex', 'Convert hexadecimal 2AF to decimal. Return only the integer.', str(int('2AF', 16))),
        ('fraction', 'Compute (3/4 + 5/6) * 24. Return only the integer.', '38'),
        ('percentage2', 'An item costs 160. Its price rises by 25 percent, then falls by 20 percent. What is its final price? Return only the integer.', '160'),
        ('gcd', 'Find the greatest common divisor of 252 and 198. Return only the integer.', '18'),
        ('median', 'Find the median of [19, 2, 13, 5, 11, 7, 3]. Return only the integer.', '7'),
        ('combinations', 'How many distinct unordered pairs can be chosen from 9 distinct objects? Return only the integer.', '36'),
        ('filter', 'Keep values divisible by 3 from [8, 9, -6, 10, 12, 0], preserving order. Return only a JSON array.', [9, -6, 12, 0]),
        ('unique', 'Remove duplicate strings from ["oak","elm","oak","pine","elm"], preserving first occurrence. Return only a JSON array.', ['oak', 'elm', 'pine']),
        ('json2', 'Records: [{"id":"A","value":9},{"id":"B","value":4},{"id":"C","value":7}]. Return a JSON object with exactly one key, "id", identifying the record with the smallest value.', {'id': 'B'}),
        ('json3', 'Count each character in "banana". Return only a JSON object mapping characters to counts.', {'b': 1, 'a': 3, 'n': 2}),
        ('python2', 'What does this Python expression evaluate to: [x for x in range(10) if x % 3 == 1]? Return only a JSON array.', [1, 4, 7]),
        ('python3', 'What does this Python expression evaluate to: sum(i for i in range(20) if i % 4 == 0)? Return only the integer.', '40'),
        ('python4', 'What does this Python expression evaluate to: len(set("abracadabra"))? Return only the integer.', '5'),
        ('python5', 'What does this Python expression evaluate to: "abcdef"[1:5:2]? Return only the resulting letters, without quotes.', 'bd'),
        ('python6', 'What does this Python expression evaluate to: sorted(["bbb", "a", "cc"], key=len)? Return only a JSON array.', ['a', 'cc', 'bbb']),
        ('python7', 'Python executes: a=[1,2]; b=a; b.append(3). What is a now? Return only a JSON array.', [1, 2, 3]),
        ('extract2', 'Records: A{owner=Luis;code=R8D2}, B{owner=Nora;code=K3P7}, C{owner=Inez;code=T6W1}. Return only the code owned by Nora.', 'K3P7'),
        ('extract3', 'The old deployment identifier was app-741. It was replaced by app-926. The rollback target remains app-741. Return only the current deployment identifier.', 'app-926'),
        ('order2', 'A finishes before C. B finishes after C. D finishes before A. Return all four letters in finishing order as a JSON array.', ['D', 'A', 'C', 'B']),
        ('logic', 'All zorbs are blue. No blue object is square. Can a zorb be square? Return only yes or no.', 'no'),
        ('stack', 'A stack starts empty. Push A, push B, pop, push C, push D, pop. What is now on top? Return only its letter.', 'C'),
        ('queue', 'A FIFO queue starts empty. Enqueue A, enqueue B, dequeue, enqueue C. What is removed by the next dequeue? Return only its letter.', 'B'),
    ]
    topics = ['how DNS resolution and negative caching interact',
              'how operating systems handle copy-on-write memory',
              'how database indexes affect range queries and writes',
              'how OAuth authorization-code flow uses PKCE',
              'how erasure coding reconstructs missing storage shards',
              'how distributed systems handle clock skew']
    base = {'context': 8192, 'mode': 'chat', 'temperature': 0, 'output_cap': 128,
            'check_output_cap': 128, 'purpose': 'Additional fixed workload; not a broad capability benchmark'}
    short = dict(base, checks=[case(*c) for c in checks], speed=[
        case(f'new-{i}', f'Explain {topic}. Include concrete worked examples and discuss limitations. Write at least 1000 words.')
        for i, topic in enumerate(topics)])
    long = dict(base, checks=[], speed=[])
    # ExLlama's job scheduler reserves max_new_tokens + 1 even without a draft
    # model. Leave that position free in both runtimes' matched 8K fixture.
    for target in (2048, 8063):
        notes = []
        index = 0
        while True:
            notes.append(f'Note {index}: a worker accepts a job, writes a durable record, retries after a timeout, and records completion. The database assigns an increasing sequence number.\n')
            text = 'Review these system-design notes:\n' + ''.join(notes)
            question = '\nExplain the failure cases and propose improvements, with worked examples. Write at least 1000 words.'
            trial = case(f'context-{target}', text + question)
            if len(trial['input_ids']) > target:
                notes.pop()
                break
            index += 1
        text = 'Review these system-design notes:\n' + ''.join(notes)
        trial = case(f'context-{target}', text + question)
        # Each appended " x" is one token with this tokenizer. Assert rather than
        # silently accepting a different context length or truncating chat markers.
        padding = target - len(trial['input_ids'])
        for _ in range(8):
            trial = case(f'context-{target}', text + ' x' * padding + question)
            difference = target - len(trial['input_ids'])
            if not difference:
                break
            padding += difference
            assert padding >= 0
        assert len(trial['input_ids']) == target
        long['speed'].append(trial)
    for name, fixture in [('additional-short.json', short), ('long-inputs.json', long)]:
        path = args.output / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + '\n')
        print(name, {'checks': len(fixture['checks']), 'input_tokens': [len(c['input_ids']) for c in fixture['speed']]})


if __name__ == '__main__':
    main()

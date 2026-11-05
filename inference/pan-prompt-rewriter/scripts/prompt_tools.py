#!/usr/bin/env python3
"""Deterministic assembly and fail-closed validation; no model or inference calls."""
import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'assets/generic_negative.txt'
TEMPLATE_SHA256 = '280152759e5488f547da6ac13f6a72ebe05fdf959371a6571c7e9782d466c501'
POSITIVE_SOFT_LIMIT = 1480
HARD_LIMIT = 1536
KEYS = ['video_style', 'lighting', 'perspective_view', 'subjects_lifecycle',
        'init_frame_visual_description', 'events', 'quality']
CAMERAS = {'static', 'pan_left', 'pan_right', 'tilt_up', 'tilt_down',
           'track_left', 'track_right', 'dolly_forward', 'dolly_backward',
           'dolly_up', 'dolly_down', 'zoom_in', 'zoom_out',
           'roll_clockwise', 'roll_counterclockwise', 'unstable'}
TIME_PATHS = {'/events/0/time_range_sec', '/events/0/camera_motion/0/time_range_sec',
              '/events/0/contained_actions/0/time_range_sec'}
ID = re.compile(r'<\|Subject[1-9][0-9]*\|>')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key: ' + key)
        out[key] = value
    return out

def loads(raw):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid)

def read(path):
    return loads(Path(path).read_bytes())

def save_new(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as out:
        out.write(text)

def duration_value(value):
    require(type(value) in (int, float), 'Duration must be a JSON number, not a string/bool')
    d = Decimal(str(value))
    require(d.is_finite() and d > 0 and d * 10 == (d * 10).to_integral_value(),
            'Duration must be positive, finite, with at most one decimal place')
    return d

def normalize_request(request):
    require(type(request) is dict, 'Request must be an object')
    require(type(request.get('original_prompt')) is str and request['original_prompt'].strip(),
            'original_prompt must be a nonempty string')
    require(not {'refined_prompt', 'negative_prompt'} & request.keys(),
            'Request must not contain existing captions; keep saved positives separate')
    row = deepcopy(request)
    values = [duration_value(row[k]) for k in ('duration_sec', 'target_duration_sec') if k in row]
    require(not values or all(x == values[0] for x in values), 'Conflicting duration aliases')
    d = values[0] if values else Decimal(13)
    if not values:
        row['duration_sec'] = 13
    return row, d

def time_paths(value, prefix=''):
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if key == 'time_range_sec':
                result[prefix + '/' + key] = item
            else:
                result.update(time_paths(item, prefix + '/' + key))
        return result
    if type(value) is list:
        return {k: v for i, item in enumerate(value)
                for k, v in time_paths(item, prefix + '/' + str(i)).items()}
    return {}

def negative_for_duration(value):
    d = duration_value(value)
    raw = TEMPLATE.read_bytes()
    require(sha(raw) == TEMPLATE_SHA256, 'Bundled fixed negative template changed')
    text = raw.decode('utf-8').strip()
    ordered(loads(text), KEYS, 'Negative template')
    require(time_paths(loads(text)) == dict.fromkeys(TIME_PATHS, '0 to {duration_sec}'),
            'Unexpected fixed negative timestamp layout')
    end = format(d.normalize(), 'f')
    result, n = re.subn(r'("time_range_sec"\s*:\s*")0 to \{duration_sec\}(")',
                        lambda m: m[1] + '0 to ' + end + m[2], text)
    require(n == 3, 'Expected exactly three timestamp replacements')
    require(time_paths(loads(result)) == dict.fromkeys(TIME_PATHS, '0 to ' + end),
            'Negative duration coverage changed')
    require(result.replace('"0 to ' + end + '"', '"0 to {duration_sec}"') == text,
            'Negative changed outside timestamps')
    return result

def ordered(value, keys, label):
    require(type(value) is dict and list(value) == keys, label + ': wrong keys/order')

def string(value, label):
    require(type(value) is str and bool(value.strip()), label + ': expected nonempty text')

def interval(value):
    require(type(value) is str and re.fullmatch(r'\d+(?:\.\d)? to \d+(?:\.\d)?', value),
            'Invalid time interval')
    start, end = map(Decimal, value.split(' to '))
    require(0 <= start < end, 'Interval must have positive duration')
    return start, end

def positive_structure(text, duration):
    string(text, 'refined_prompt')
    positive = loads(text)
    ordered(positive, KEYS, 'Positive')
    for key in ('video_style', 'lighting', 'perspective_view', 'init_frame_visual_description'):
        string(positive[key], key)
        require(not ID.search(positive[key]), 'Subject IDs only belong in declarations/action strings')
    require(len(positive['init_frame_visual_description'].split()) <= 200, 'Initial description exceeds 200 words')
    subjects = positive['subjects_lifecycle']
    require(type(subjects) is list, 'subjects_lifecycle must be an array')
    ids = set()
    for i, subject in enumerate(subjects, 1):
        ordered(subject, ['subject_id', 'display_name', 'subject_attribute'], 'Subject')
        require(subject['subject_id'] == '<|Subject' + str(i) + '|>', 'Subject IDs must be sequential')
        for key in ('display_name', 'subject_attribute'):
            string(subject[key], key)
            require(not ID.search(subject[key]), 'Unexpected subject token in appearance')
        ids.add(subject['subject_id'])
    events = positive['events']
    require(type(events) is list and bool(events), 'Positive events must be nonempty')
    current = Decimal(0)
    for i, event in enumerate(events):
        ordered(event, ['event_index', 'time_range_sec', 'event_description', 'camera_motion', 'contained_actions'], 'Event')
        require(type(event['event_index']) is int and event['event_index'] == i, 'Event indices must be sequential')
        start, end = interval(event['time_range_sec'])
        require(start == current and end <= duration, 'Events must cover duration contiguously')
        current = end
        string(event['event_description'], 'event_description')
        require(len(event['event_description'].split()) <= 200, 'Event description exceeds 200 words')
        require(not ID.search(event['event_description']), 'No subject IDs in event prose')
        camera = event['camera_motion']
        require(type(camera) is list and bool(camera), 'Camera coverage must be nonempty')
        position = start
        for row in camera:
            ordered(row, ['time_range_sec', 'camera_motion'], 'Camera entry')
            a, b = interval(row['time_range_sec'])
            require(a == position and b <= end, 'Camera must cover parent event contiguously')
            position = b
            tags = row['camera_motion']
            require(type(tags) is list and bool(tags) and all(type(t) is str and t in CAMERAS for t in tags),
                    'Unknown positive camera tag')
            require(len(set(tags)) == len(tags) and ('static' not in tags or len(tags) == 1),
                    'Duplicate/conflicting camera tags')
        require(position == end, 'Incomplete camera coverage')
        require(type(event['contained_actions']) is list, 'Actions must be an array')
        for action in event['contained_actions']:
            ordered(action, ['time_range_sec', 'action'], 'Action')
            a, b = interval(action['time_range_sec'])
            require(start <= a < b <= end, 'Action outside parent event')
            string(action['action'], 'action')
            tokens = ID.findall(action['action'])
            require(tokens and action['action'].startswith(tokens[0]) and set(tokens) <= ids,
                    'Action needs a registered performer and registered subject references')
    require(current == duration, 'Positive does not cover exact requested duration')
    quality = positive['quality']
    ordered(quality, ['video_quality', 'text_logos_overlays', 'artifacts'], 'Quality')
    string(quality['video_quality'], 'video_quality')
    require(quality['artifacts'] is None, 'Positive artifacts must be null')
    require(quality['text_logos_overlays'] is None or type(quality['text_logos_overlays']) is str,
            'Positive text_logos_overlays must be string/null')
    require(not ID.search(json.dumps(quality)), 'No subject IDs in quality')
    return positive

def collection(value):
    if type(value) is list:
        require(bool(value), 'Empty batches are not supported')
        return value
    require(type(value) is dict, 'Expected object or nonempty list')
    return [value]

def assemble(request, positive):
    require(type(request) is type(positive), 'Request/positive collection shapes differ')
    requests, positives = collection(request), collection(positive)
    require(len(requests) == len(positives), 'Request/positive batch lengths differ')
    output = []
    for raw, authored in zip(requests, positives):
        ordered(authored, ['refined_prompt'], 'Authored positive response')
        row, d = normalize_request(raw)
        positive_structure(authored['refined_prompt'], d)
        row['refined_prompt'] = authored['refined_prompt']
        row['negative_prompt'] = negative_for_duration(float(d))
        require(list(loads(row['refined_prompt'])) == list(loads(row['negative_prompt'])) == KEYS,
                'Positive/negative field-order mismatch')
        output.append(row)
    return output if type(request) is list else output[0]

def validate(args):
    paths = [args.request, args.positive, args.candidate]
    require(not args.report.exists(), 'Report already exists; use a new revision path')
    original = {str(p.resolve()): p.read_bytes() for p in paths}
    receipt = dict(ok=False, preservation_ok=False, tokenizer_ok=False, errors=[],
                   input_sha256={p: sha(v) for p, v in original.items()},
                   template_sha256=sha(TEMPLATE.read_bytes()),
                   system_prompt_sha256=sha((ROOT / 'system_prompt.md').read_bytes()),
                   positive_soft_limit=POSITIVE_SOFT_LIMIT, hard_limit=HARD_LIMIT)
    code = 1
    try:
        expected = assemble(read(args.request), read(args.positive))
        candidate = read(args.candidate)
        require(canonical(candidate) == canonical(expected), 'Candidate differs from exact assembly')
        receipt['preservation_ok'] = True
        if args.tokenizer is None:
            receipt['errors'].append('Unverified draft — production encoder-token counts have not been verified.')
            code = 2
        else:
            from production_tokens import EncoderTokenCounter
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
            tokenizer = args.tokenizer.resolve(strict=True)
            counter = EncoderTokenCounter(str(tokenizer))
            checks = []
            for i, row in enumerate(collection(candidate)):
                p, n = counter(row['refined_prompt']), counter(row['negative_prompt'])
                checks.append(dict(index=i, positive_tokens=p, negative_tokens=n,
                                   positive_soft_limit_ok=p <= POSITIVE_SOFT_LIMIT,
                                   hard_limit_ok=max(p,n) <= HARD_LIMIT))
            receipt.update(tokenizer_ok=True, checks=checks, tokenizer_path=str(tokenizer),
                           counter_sha256=sha((ROOT / 'scripts/production_tokens.py').read_bytes()),
                           tokenizer_assets={p.relative_to(tokenizer).as_posix(): sha(p.read_bytes())
                             for p in sorted(tokenizer.rglob('*')) if p.is_file() and p.suffix in ('.json','.jinja','.model','.txt')})
            receipt['ok'] = all(c['positive_soft_limit_ok'] and c['hard_limit_ok'] for c in checks)
            if not receipt['ok']:
                receipt['errors'].append('Token budget exceeded; revise only the positive when needed')
            code = 0 if receipt['ok'] else 1
    except Exception as exc:
        receipt['errors'].append(str(exc))
    receipt['inputs_unchanged'] = all(Path(p).read_bytes() == raw for p, raw in original.items())
    if not receipt['inputs_unchanged']:
        receipt['ok'] = False
        receipt['errors'].append('Input changed during validation')
        code = 1
    save_new(args.report, receipt)
    print(json.dumps({k:receipt[k] for k in ('ok','preservation_ok','tokenizer_ok','errors')}))
    return code


def main():
    parser = argparse.ArgumentParser(description='Assemble and validate caption pairs')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('assemble','validate'):
        p = sub.add_parser(name)
        p.add_argument('--request', type=Path, required=True)
        p.add_argument('--positive', type=Path, required=True)
        if name == 'assemble':
            p.add_argument('--output', type=Path, required=True)
        else:
            p.add_argument('--candidate', type=Path, required=True)
            p.add_argument('--report', type=Path, required=True)
            p.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'assemble':
            save_new(args.output, assemble(read(args.request), read(args.positive)))
            print(json.dumps({'assembled':str(args.output),'token_verified':False}))
            return 0
        return validate(args)
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)}))
        return 1

if __name__ == '__main__':
    raise SystemExit(main())

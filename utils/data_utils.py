import difflib
import javalang
import re
import json
import torch
from typing import List, NamedTuple, Dict
from utils.diff_utils import INSERT, INSERT_END, UPDATEFROM, UPDATETO, UPDATE_END, DEL, DEL_END, KEEP, KEEP_END


class GenerationBatchData(NamedTuple):
    """Stores tensorized batch used in generation model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]

class UpdateBatchData(NamedTuple):
    """Stores tensorized batch used in edit model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    old_nl_ids: torch.Tensor
    old_nl_lengths: torch.Tensor
    ast_ids: torch.Tensor
    ast_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]
    code_features: torch.Tensor
    nl_features: torch.Tensor
    ast_features: torch.Tensor

class Example(NamedTuple):
    id: str
    old_comment: str
    old_comment_tokens: List[str]
    new_comment: str
    new_comment_tokens: List[str]
    old_code: str
    old_code_tokens: List[str]
    new_code: str
    new_code_tokens: List[str]

class DiffExample(NamedTuple):
    id: str
    old_comment: str
    old_comment_tokens: List[str]
    new_comment: str
    new_comment_tokens: List[str]
    old_code: str
    old_code_tokens: List[str]
    new_code: str
    new_code_tokens: List[str]
    span_diff_code: str
    span_diff_code_tokens: List[str]
    span_minimal_diff_comment: str
    span_minimal_diff_comment_tokens: List[str]
    token_diff_code_tokens: List[str]
    ast_diff: Dict[str, List[str]]
    ast_diff_tokens: List[str]
    variables: List[str]
    dependency: List[List[int]]

def read_examples_from_file(filename):
    data = []
    with open(filename) as f:
        for js in f.readlines():
            data.append(json.loads(js))
    return [DiffExample(**d) for d in data]


SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']


def subtokenize_comment(comment_line):
    comment_line = remove_return_string(comment_line)
    comment_line = remove_html_tag(
        comment_line.replace('/**', '').replace('**/', '').replace('/*', '').replace('*/', '').replace('*', '').strip())
    comment_line = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip())
    comment_line = ' '.join(comment_line)
    comment_line = comment_line.replace('\n', ' ').strip()

    tokens = comment_line.split(' ')
    subtokens = []
    labels = []
    indices = []

    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            labels.append(0)
            indices.append(0)
            subtokens.append(curr[0].lower())
            continue

        for s, subtoken in enumerate(curr):
            labels.append(1)
            indices.append(s)
            subtokens.append(curr[s].lower())

    return subtokens, labels, indices


def subtokenize_code(line):
    try:
        tokens = get_clean_code(list(javalang.tokenizer.tokenize(line)))
    except:
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", line.strip())
    subtokens = []
    labels = []
    indices = []
    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            labels.append(0)
            indices.append(0)
            subtokens.append(curr[0].lower())
            continue
        for s, subtoken in enumerate(curr):
            labels.append(1)
            indices.append(s)
            subtokens.append(curr[s].lower())

    return subtokens, labels, indices


def subtokenize_ast(ast_list):
    add_set = set()
    delete_set = set()
    move_set = set()
    update_set = set()
    tokens = []
    will_process_type = ["SimpleName", "NullLiteral", "ReturnStatement"]
    for tup in ast_list:
        if tup[1] in will_process_type or tup[2] != "null":
            pass
        else:
            continue
        try:
            cur_tup_tokens = get_clean_code(list(javalang.tokenizer.tokenize(tup[2])))
        except:
            cur_tup_tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", tup[2].strip())
        tokens.extend(cur_tup_tokens)
        for token in cur_tup_tokens:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            curr = [item.lower() for item in curr]
            if tup[0] == 'add':
                add_set = add_set.union(set(curr))
            elif tup[0] == 'delete':
                delete_set = delete_set.union(set(curr))
            elif tup[0] == 'move':
                move_set = move_set.union(set(curr))
            elif tup[0] == 'update':
                update_set = update_set.union(set(curr))
    subtokens = []
    labels = []
    indices = []
    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            labels.append(0)
            indices.append(0)
            subtokens.append(curr[0].lower())
            continue
        for s, subtoken in enumerate(curr):
            labels.append(1)
            indices.append(s)
            subtokens.append(curr[s].lower())
    return subtokens, labels, indices, add_set, delete_set, move_set, update_set


def remove_html_tag(line):
    """Helper method for subtokenizing comment."""
    clean = re.compile('<.*?>')
    line = re.sub(clean, '', line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, '')

    return line


def remove_return_string(line):
    return line.replace('@return', '').replace('@ return', '').strip()


def get_clean_code(tokenized_code):
    token_vals = [t.value for t in tokenized_code]
    new_token_vals = []
    for t in token_vals:
        n = [c for c in re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]",
                                   t.encode('ascii', errors='ignore').decode().strip()) if len(c) > 0]
        new_token_vals = new_token_vals + n

    token_vals = new_token_vals
    cleaned_code_tokens = []

    for c in token_vals:
        try:
            cleaned_code_tokens.append(str(c))
        except:
            pass

    return cleaned_code_tokens


def compute_code_diff_spans(old_tokens, old_labels, old_indices, new_tokens, new_labels, new_indices):
    spans = []
    labels = []
    indices = []

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens,
                                                                             new_tokens).get_opcodes():
        if edit_type == 'equal':
            spans.extend([KEEP] + old_tokens[o_start:o_end] + [KEEP_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0])
        elif edit_type == 'replace':
            spans.extend(
                [UPDATEFROM] + old_tokens[o_start:o_end] + [UPDATETO] + new_tokens[n_start:n_end] + [UPDATE_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0] + new_labels[n_start:n_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0] + new_indices[n_start:n_end] + [0])
        elif edit_type == 'insert':
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
            labels.extend([0] + new_labels[n_start:n_end] + [0])
            indices.extend([0] + new_indices[n_start:n_end] + [0])
        else:
            spans.extend([DEL] + old_tokens[o_start:o_end] + [DEL_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0])

    return spans, labels, indices

